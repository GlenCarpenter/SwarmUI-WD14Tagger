"""WD14-family ONNX inference helpers."""

import csv
import os

import numpy as np
from PIL import Image

from .common import IMAGE_SIZE, ensure_hf_files, load_image_rgb, log_info, onnx_providers, remove_underscore


def preprocess_wd14_image(image_path: str) -> np.ndarray:
    """Preprocess an image for standard WD14 NHWC/BGR ONNX models."""
    image = load_image_rgb(image_path)
    img = np.array(image)[:, :, ::-1]
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
    img_pil = Image.fromarray(img[:, :, ::-1]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return np.array(img_pil)[:, :, ::-1].astype(np.float32)


def ensure_wd14_model(repo_id: str, model_dir: str) -> str:
    """Download the standard WD14 ONNX model assets when missing."""
    return ensure_hf_files(repo_id, model_dir, ["model.onnx", "selected_tags.csv"], f"Downloading model {repo_id}...")


def load_wd14_tags(model_path: str) -> tuple[list[str], list[str], int]:
    """Load general/character tag groups from a WD14-style selected_tags.csv file."""
    csv_path = os.path.join(model_path, "selected_tags.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    header = all_rows[0] if all_rows else []
    rows = all_rows[1:]
    header_lower = [h.lower().strip() for h in header]
    try:
        name_idx = header_lower.index("name")
        cat_idx = header_lower.index("category")
    except ValueError:
        name_idx, cat_idx = 1, 2

    log_info(f"CSV: header={header}, {len(rows)} data rows, name_col={name_idx}, category_col={cat_idx}")
    rating_count = sum(1 for row in rows if len(row) > cat_idx and row[cat_idx] == "9")
    general_tags = [row[name_idx] for row in rows if len(row) > cat_idx and row[cat_idx] == "0"]
    character_tags = [row[name_idx] for row in rows if len(row) > cat_idx and row[cat_idx] == "4"]
    return general_tags, character_tags, rating_count


def run_wd14_inference(image_path: str, repo_id: str, model_dir: str, general_threshold: float, character_threshold: float) -> str:
    """Run a standard WD14 ONNX model and return comma-separated tags."""
    import onnxruntime as ort

    log_info(f"Starting WD14 inference: model={repo_id}, general_threshold={general_threshold}, character_threshold={character_threshold}")
    model_path = ensure_wd14_model(repo_id, model_dir)
    providers = onnx_providers()
    log_info(f"Using providers: {providers}")

    sess = ort.InferenceSession(os.path.join(model_path, "model.onnx"), providers=providers)
    input_info = sess.get_inputs()[0]
    log_info(f"Model input: name={input_info.name}, shape={input_info.shape}, dtype={input_info.type}")
    img_batch = np.expand_dims(preprocess_wd14_image(image_path), 0)
    log_info(f"Input batch shape: {img_batch.shape}, dtype={img_batch.dtype}, min={float(img_batch.min()):.3f}, max={float(img_batch.max()):.3f}")

    output_names = [o.name for o in sess.get_outputs()]
    output_values = sess.run(output_names, {input_info.name: img_batch})
    output_map = {name: val[0] for name, val in zip(output_names, output_values)}
    probs = output_map.get("prediction", output_values[0][0])

    general_tags, character_tags, rating_count = load_wd14_tags(model_path)
    log_info(f"Tags loaded: {len(general_tags)} general, {len(character_tags)} character, {rating_count} rating prefix")
    tags = []
    for i, prob in enumerate(probs[rating_count:]):
        if i < len(general_tags):
            if general_threshold >= 0 and prob >= general_threshold:
                tags.append(remove_underscore(general_tags[i]))
        else:
            char_idx = i - len(general_tags)
            if char_idx < len(character_tags) and character_threshold >= 0 and prob >= character_threshold:
                tags.append(remove_underscore(character_tags[char_idx]))

    log_info(f"Found {len(tags)} tags above threshold")
    return ", ".join(tags)
