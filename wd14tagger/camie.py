"""Camie ONNX inference helpers."""

import json
import os

import numpy as np
from PIL import Image

from .common import ensure_hf_files, load_image_rgb, log_info, onnx_providers, remove_underscore


CAMIE_MODELS = {
    "Camais03/camie-tagger": ("model_initial.onnx", "model_initial_metadata.json"),
    "Camais03/camie-tagger-v2": ("camie-tagger-v2.onnx", "camie-tagger-v2-metadata.json"),
}


def preprocess_camie_image(image_path: str, image_size: int = 512) -> np.ndarray:
    """Preprocess an image for Camie ONNX models."""
    img = load_image_rgb(image_path)
    width, height = img.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = image_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = image_size
        new_width = int(new_height * aspect_ratio)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    padded.paste(img, ((image_size - new_width) // 2, (image_size - new_height) // 2))
    arr = np.array(padded).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0)


def ensure_camie_model(repo_id: str, model_dir: str) -> tuple[str, str, str]:
    """Download the Camie ONNX model and metadata files when missing."""
    onnx_file, tags_file = CAMIE_MODELS[repo_id]
    model_path = ensure_hf_files(repo_id, model_dir, [onnx_file, tags_file], f"Checking/downloading Camie model {repo_id}...")
    return model_path, onnx_file, tags_file


def load_camie_metadata(model_path: str, tags_filename: str) -> tuple[dict, dict]:
    """Load idx_to_tag and tag_to_category from Camie metadata JSON."""
    with open(os.path.join(model_path, tags_filename), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    mapping = metadata["dataset_info"]["tag_mapping"] if "dataset_info" in metadata else metadata
    return mapping["idx_to_tag"], mapping["tag_to_category"]


def run_camie_inference(image_path: str, repo_id: str, model_dir: str, general_threshold: float, character_threshold: float) -> str:
    """Run Camie ONNX inference and return comma-separated tags."""
    import onnxruntime as ort

    log_info(f"Starting Camie inference: model={repo_id}, general_threshold={general_threshold}, character_threshold={character_threshold}")
    model_path, onnx_file, tags_file = ensure_camie_model(repo_id, model_dir)
    providers = onnx_providers()
    log_info(f"Using providers: {providers}")

    sess = ort.InferenceSession(os.path.join(model_path, onnx_file), providers=providers)
    input_info = sess.get_inputs()[0]
    img_batch = preprocess_camie_image(image_path)
    if input_info.type == "tensor(float)":
        img_batch = img_batch.astype(np.float32)
    outputs = sess.run(None, {input_info.name: img_batch})
    raw = outputs[1] if len(outputs) > 1 else outputs[0]
    probs = 1.0 / (1.0 + np.exp(-raw[0]))

    idx_to_tag, tag_to_category = load_camie_metadata(model_path, tags_file)
    tags = []
    for idx_str, tag_name in idx_to_tag.items():
        idx = int(idx_str)
        if idx >= len(probs):
            continue
        prob = float(probs[idx])
        category = tag_to_category.get(tag_name, "general")
        if category == "rating":
            continue
        if category == "character":
            if character_threshold < 0 or prob < character_threshold:
                continue
        elif general_threshold < 0 or prob < general_threshold:
            continue
        tags.append(remove_underscore(tag_name))

    log_info(f"Found {len(tags)} Camie tags above threshold")
    return ", ".join(tags)
