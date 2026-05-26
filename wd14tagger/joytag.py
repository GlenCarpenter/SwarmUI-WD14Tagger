"""JoyTag ONNX inference helpers."""

import os

import numpy as np
from PIL import Image

from .common import ensure_hf_files, load_image_rgb, log_info, onnx_providers, remove_underscore


JOYTAG_REPO_ID = "fancyfeast/joytag"
JOYTAG_IMAGE_SIZE = 448
JOYTAG_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
JOYTAG_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_joytag_image(image_path: str) -> np.ndarray:
    """Preprocess an image for JoyTag's NCHW/RGB ONNX model."""
    image = load_image_rgb(image_path)
    width, height = image.size
    max_dim = max(width, height)
    pad_left = (max_dim - width) // 2
    pad_top = (max_dim - height) // 2

    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, (pad_left, pad_top))
    if max_dim != JOYTAG_IMAGE_SIZE:
        padded = padded.resize((JOYTAG_IMAGE_SIZE, JOYTAG_IMAGE_SIZE), Image.BICUBIC)

    img = np.array(padded).astype(np.float32) / 255.0
    img = (img - JOYTAG_MEAN) / JOYTAG_STD
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)


def ensure_joytag_model(repo_id: str, model_dir: str) -> str:
    """Download the JoyTag ONNX model assets when missing."""
    return ensure_hf_files(repo_id, model_dir, ["model.onnx", "top_tags.txt"], f"Downloading JoyTag model {repo_id}...")


def load_joytag_tags(model_path: str) -> list[str]:
    """Load JoyTag's ordered tag list."""
    with open(os.path.join(model_path, "top_tags.txt"), "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def pick_joytag_threshold(general_threshold: float, character_threshold: float) -> float:
    """Pick the threshold JoyTag should use with SwarmUI's existing threshold UI."""
    if general_threshold >= 0:
        return general_threshold
    if character_threshold >= 0:
        return character_threshold
    return -1.0


def run_joytag_inference(image_path: str, repo_id: str, model_dir: str, general_threshold: float, character_threshold: float) -> str:
    """Run JoyTag ONNX inference and return comma-separated tags."""
    import onnxruntime as ort

    threshold = pick_joytag_threshold(general_threshold, character_threshold)
    if threshold < 0:
        log_info("JoyTag thresholds are both disabled; returning no tags")
        return ""

    log_info(
        f"Starting JoyTag inference: model={repo_id}, general_threshold={general_threshold}, character_threshold={character_threshold}, effective_threshold={threshold}"
    )
    model_path = ensure_joytag_model(repo_id, model_dir)
    providers = onnx_providers()
    log_info(f"Using providers: {providers}")

    sess = ort.InferenceSession(os.path.join(model_path, "model.onnx"), providers=providers)
    input_info = sess.get_inputs()[0]
    img_batch = preprocess_joytag_image(image_path)
    log_info(f"JoyTag input batch shape: {img_batch.shape}, dtype={img_batch.dtype}, min={float(img_batch.min()):.3f}, max={float(img_batch.max()):.3f}")

    raw_scores = sess.run(None, {input_info.name: img_batch})[0][0]
    probs = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -80.0, 80.0)))
    top_tags = load_joytag_tags(model_path)

    tags = []
    for idx, tag_name in enumerate(top_tags):
        if idx >= len(probs):
            break
        if float(probs[idx]) >= threshold:
            tags.append(remove_underscore(tag_name))

    log_info(f"Found {len(tags)} JoyTag tags above threshold")
    return ", ".join(tags)