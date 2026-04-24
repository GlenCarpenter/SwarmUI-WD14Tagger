#!/usr/bin/env python3
"""
WD14 Tagger inference script for SwarmUI WD14TaggerExtension.
Standalone ONNX-based tagger using SmilingWolf models from HuggingFace.
Outputs a single JSON line to stdout: {"success": true, "tags": "..."} or {"success": false, "error": "..."}
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

IMAGE_SIZE = 448


def preprocess_image(image_path: str, bgr: bool = True) -> np.ndarray:
    """Load and preprocess an image for the WD14 tagger ONNX model.

    When bgr=True (SmilingWolf/NHWC models): pads to square, resizes with BICUBIC, returns BGR float32 0-255.
    When bgr=False (PixAI/NCHW models): resizes directly with BILINEAR, returns RGB float32 0-255.
    """
    image = Image.open(image_path)

    # Handle transparency with white background compositing
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")

    if bgr:
        img = np.array(image)
        img = img[:, :, ::-1]  # RGB -> BGR

        # Pad shorter side to make a square
        h, w = img.shape[:2]
        size = max(h, w)
        pad_y = size - h
        pad_x = size - w
        img = np.pad(
            img,
            ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        # Resize to model input size (BICUBIC, back to RGB for PIL then back to BGR)
        img_pil = Image.fromarray(img[:, :, ::-1]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        img = np.array(img_pil)[:, :, ::-1].astype(np.float32)
    else:
        # Direct resize to model input size without BGR conversion (NCHW models use RGB)
        img_pil = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img = np.array(img_pil).astype(np.float32)

    return img


def ensure_model(repo_id: str, model_dir: str) -> str:
    """Download model.onnx and selected_tags.csv from HuggingFace if not cached locally."""
    safe_name = repo_id.replace("/", "_")
    model_path = os.path.join(model_dir, safe_name)
    onnx_file = os.path.join(model_path, "model.onnx")
    csv_file = os.path.join(model_path, "selected_tags.csv")

    if not os.path.exists(onnx_file) or not os.path.exists(csv_file):
        os.makedirs(model_path, exist_ok=True)
        print(json.dumps({"progress": f"Downloading model {repo_id}..."}), flush=True)
        hf_hub_download(repo_id=repo_id, filename="model.onnx", local_dir=model_path)
        hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", local_dir=model_path)

    return model_path


def load_tags(model_path: str):
    """Read general and character tag lists from selected_tags.csv (skip header row).
    Also returns the number of leading rating predictions (category 9) in the model output."""
    csv_path = os.path.join(model_path, "selected_tags.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    header = all_rows[0] if all_rows else []
    rows = all_rows[1:]

    # Detect column indices from header (SmilingWolf: tag_id,name,category,count;
    # PixAI: id,tag_id,name,category,count,ips)
    header_lower = [h.lower().strip() for h in header]
    try:
        name_idx = header_lower.index("name")
        cat_idx = header_lower.index("category")
    except ValueError:
        # Fallback to legacy SmilingWolf positional layout
        name_idx, cat_idx = 1, 2

    log_info(f"CSV: header={header}, {len(rows)} data rows, name_col={name_idx}, category_col={cat_idx}")

    rating_count = sum(1 for row in rows if len(row) > cat_idx and row[cat_idx] == "9")
    general_tags = [row[name_idx] for row in rows if len(row) > cat_idx and row[cat_idx] == "0"]
    character_tags = [row[name_idx] for row in rows if len(row) > cat_idx and row[cat_idx] == "4"]
    return general_tags, character_tags, rating_count


def remove_underscore(tag: str) -> str:
    """Replace underscores with spaces except for very short tags (<=3 chars)."""
    return tag.replace("_", " ") if len(tag) > 3 else tag


def log_info(msg: str):
    """Emit an info log line to stdout for the C# host to forward."""
    print(json.dumps({"info": msg}), flush=True)


def log_error(msg: str):
    """Emit an error log line to stdout for the C# host to forward."""
    print(json.dumps({"error": msg}), flush=True)


def run_inference(image_path: str, repo_id: str, model_dir: str, threshold: float) -> str:
    """Run WD14 tagger inference and return comma-separated tag string."""
    import onnxruntime as ort

    log_info(f"Starting inference: model={repo_id}, threshold={threshold}")

    model_path = ensure_model(repo_id, model_dir)
    onnx_path = os.path.join(model_path, "model.onnx")
    log_info(f"Model path: {model_path}")

    # Pick best available execution provider
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "ROCMExecutionProvider" in available:
        providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    log_info(f"Using providers: {providers}")

    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_info = sess.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    input_dtype = input_info.type
    log_info(f"Model input: name={input_name}, shape={input_shape}, dtype={input_dtype}")

    # Detect channel order from model input shape: NCHW has shape[1]==3, NHWC has shape[3]==3
    if len(input_shape) == 4 and input_shape[1] == 3:
        # NCHW model (e.g. PixAI): RGB input, normalized to [-1, 1], channels-first
        log_info("Detected NCHW layout — using RGB preprocessing with [-1,1] normalization")
        img = preprocess_image(image_path, bgr=False)
        img = (img / 255.0 - 0.5) / 0.5
        img_batch = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
    else:
        # NHWC model (SmilingWolf): BGR input, 0-255 range, channels-last
        log_info("Detected NHWC layout — using BGR preprocessing with 0-255 range")
        img = preprocess_image(image_path)
        img_batch = np.expand_dims(img, 0)
    log_info(f"Input batch shape: {img_batch.shape}, dtype={img_batch.dtype}, min={float(img_batch.min()):.3f}, max={float(img_batch.max()):.3f}")

    output_infos = sess.get_outputs()
    log_info(f"Model outputs: {[(o.name, o.shape) for o in output_infos]}")
    output_names = [o.name for o in output_infos]
    output_values = sess.run(output_names, {input_name: img_batch})
    output_map = {name: val[0] for name, val in zip(output_names, output_values)}

    # Prefer the 'prediction' output (tag probabilities); fall back to first output
    if "prediction" in output_map:
        probs = output_map["prediction"]
        log_info(f"Using 'prediction' output: {len(probs)} values")
    else:
        probs = output_values[0][0]
        log_info(f"No 'prediction' output found, using first output: {len(probs)} values")
    log_info(f"Inference complete: {len(probs)} probabilities, max={float(probs.max()):.4f}, mean={float(probs.mean()):.4f}")

    general_tags, character_tags, rating_count = load_tags(model_path)
    log_info(f"Tags loaded: {len(general_tags)} general, {len(character_tags)} character, {rating_count} rating prefix")

    tags = []
    # probs[0:rating_count] are rating predictions (ignored for output)
    # probs[rating_count:] map to general_tags then character_tags in order
    for i, p in enumerate(probs[rating_count:]):
        if i < len(general_tags):
            if p >= threshold:
                tags.append(remove_underscore(general_tags[i]))
        else:
            char_idx = i - len(general_tags)
            if char_idx < len(character_tags) and p >= threshold:
                tags.append(remove_underscore(character_tags[char_idx]))

    log_info(f"Found {len(tags)} tags above threshold {threshold}")
    return ", ".join(tags)


def main():
    parser = argparse.ArgumentParser(description="WD14 Tagger inference (ONNX, SwarmUI extension)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="SmilingWolf/wd-eva02-large-tagger-v3",
        help="HuggingFace model repo ID (e.g. SmilingWolf/wd-eva02-large-tagger-v3)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="Models/wd14_tagger",
        help="Local directory to store downloaded models",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for including a tag (0.0-1.0)",
    )
    args = parser.parse_args()

    try:
        tags = run_inference(args.image_path, args.repo_id, args.model_dir, args.threshold)
        print(json.dumps({"success": True, "tags": tags}), flush=True)
    except Exception as e:
        import traceback
        print(json.dumps({"error": traceback.format_exc()}), flush=True)
        print(json.dumps({"success": False, "error": str(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
