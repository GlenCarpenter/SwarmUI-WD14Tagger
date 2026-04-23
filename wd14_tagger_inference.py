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
# All SmilingWolf default-format models lead with 4 rating predictions
RATING_COUNT = 4


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess an image for the WD14 tagger ONNX model."""
    image = Image.open(image_path)

    # Handle transparency with white background compositing
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")

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
    """Read general and character tag lists from selected_tags.csv (skip header row)."""
    csv_path = os.path.join(model_path, "selected_tags.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)[1:]  # skip header: tag_id,name,category,count

    general_tags = [row[1] for row in rows if row[2] == "0"]
    character_tags = [row[1] for row in rows if row[2] == "4"]
    return general_tags, character_tags


def remove_underscore(tag: str) -> str:
    """Replace underscores with spaces except for very short tags (<=3 chars)."""
    return tag.replace("_", " ") if len(tag) > 3 else tag


def run_inference(image_path: str, repo_id: str, model_dir: str, threshold: float) -> str:
    """Run WD14 tagger inference and return comma-separated tag string."""
    import onnxruntime as ort

    model_path = ensure_model(repo_id, model_dir)
    onnx_path = os.path.join(model_path, "model.onnx")

    # Pick best available execution provider
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "ROCMExecutionProvider" in available:
        providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    img = preprocess_image(image_path)
    img_batch = np.expand_dims(img, 0)
    probs = sess.run(None, {input_name: img_batch})[0][0]

    general_tags, character_tags = load_tags(model_path)

    tags = []
    # probs[0:RATING_COUNT] are rating predictions (ignored for output)
    # probs[RATING_COUNT:] map to general_tags then character_tags in order
    for i, p in enumerate(probs[RATING_COUNT:]):
        if i < len(general_tags):
            if p >= threshold:
                tags.append(remove_underscore(general_tags[i]))
        else:
            char_idx = i - len(general_tags)
            if char_idx < len(character_tags) and p >= threshold:
                tags.append(remove_underscore(character_tags[char_idx]))

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
        print(json.dumps({"success": True, "tags": tags}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
