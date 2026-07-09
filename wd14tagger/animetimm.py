"""AnimeTimm (deepghs) TIMM-based tagger inference helpers.

Supports the multi-label tagger models published under the HuggingFace
``animetimm`` organization (https://huggingface.co/animetimm). Each model repo
ships a self-describing preprocessing pipeline (``preprocess.json``) plus tag
and category metadata, so a single implementation handles every backbone and
input size in the family.

Most repos publish a ``model.onnx`` export that runs through onnxruntime. A few
(typically the largest backbones) only publish timm ``safetensors`` weights; for
those this module falls back to building the timm model and running it with
PyTorch, reusing the exact same preprocessing, tags, and categories.
"""

import csv
import json
import os

import numpy as np
from PIL import Image

from .common import ensure_hf_files, load_image_rgb, log_info, onnx_providers, remove_underscore


ANIMETIMM_PREFIX = "animetimm/"

ANIMETIMM_METADATA_FILES = ["selected_tags.csv", "categories.json", "preprocess.json"]

ANIMETIMM_SAFETENSORS_FILES = ["config.json", "model.safetensors"]

_INTERPOLATION_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "box": Image.BOX,
    "hamming": Image.HAMMING,
    "lanczos": Image.LANCZOS,
}


def is_animetimm_repo(repo_id: str) -> bool:
    """Return True when the repo ID belongs to the animetimm model family."""
    return repo_id.startswith(ANIMETIMM_PREFIX)


def ensure_animetimm_model(repo_id: str, model_dir: str) -> tuple[str, str]:
    """Download an animetimm model plus its metadata, preferring the ONNX export.

    Returns ``(model_path, model_type)`` where ``model_type`` is either ``"onnx"``
    or ``"safetensors"``. Not every animetimm repo ships a ``model.onnx`` export;
    repos that only publish timm weights fall back to safetensors inference.
    """
    from huggingface_hub.errors import EntryNotFoundError

    model_path = ensure_hf_files(repo_id, model_dir, ANIMETIMM_METADATA_FILES, f"Downloading animetimm metadata for {repo_id}...")
    if os.path.exists(os.path.join(model_path, "model.onnx")):
        return model_path, "onnx"
    if all(os.path.exists(os.path.join(model_path, name)) for name in ANIMETIMM_SAFETENSORS_FILES):
        return model_path, "safetensors"
    try:
        ensure_hf_files(repo_id, model_dir, ["model.onnx"], f"Downloading animetimm model {repo_id}...")
        return model_path, "onnx"
    except EntryNotFoundError:
        log_info(f"{repo_id} has no ONNX export; falling back to timm safetensors weights.")
        ensure_hf_files(repo_id, model_dir, ANIMETIMM_SAFETENSORS_FILES, f"Downloading animetimm safetensors model {repo_id}...")
        return model_path, "safetensors"


def _resolve_interpolation(value) -> int:
    """Map a preprocess.json interpolation name onto a Pillow resample constant."""
    if isinstance(value, str):
        return _INTERPOLATION_MAP.get(value.lower(), Image.BICUBIC)
    return Image.BICUBIC


def _apply_resize(image: Image.Image, step: dict) -> Image.Image:
    """Apply a dghs-imgutils 'resize' transform to a PIL image."""
    size = step.get("size")
    interpolation = _resolve_interpolation(step.get("interpolation", "bilinear"))
    max_size = step.get("max_size")
    width, height = image.size
    if isinstance(size, (list, tuple)):
        if len(size) == 1:
            size = int(size[0])
        else:
            # dghs-imgutils stores 2-element sizes as (height, width).
            target_h, target_w = int(size[0]), int(size[1])
            return image.resize((target_w, target_h), interpolation)
    short_side = int(size)
    if width < height:
        new_width = short_side
        new_height = int(short_side * height / width)
    else:
        new_height = short_side
        new_width = int(short_side * width / height)
    if max_size is not None:
        longest = max(new_width, new_height)
        if longest > max_size:
            new_width = int(max_size * new_width / longest)
            new_height = int(max_size * new_height / longest)
    if (new_width, new_height) == (width, height):
        return image
    return image.resize((new_width, new_height), interpolation)


def _apply_center_crop(image: Image.Image, step: dict) -> Image.Image:
    """Apply a dghs-imgutils 'center_crop' transform to a PIL image."""
    size = step.get("size")
    if isinstance(size, (list, tuple)):
        if len(size) == 1:
            crop_height = crop_width = int(size[0])
        else:
            # dghs-imgutils stores crop sizes as (height, width).
            crop_height, crop_width = int(size[0]), int(size[1])
    else:
        crop_height = crop_width = int(size)
    width, height = image.size
    if width < crop_width or height < crop_height:
        pad_width = max(crop_width - width, 0)
        pad_height = max(crop_height - height, 0)
        padded = Image.new(image.mode, (width + pad_width, height + pad_height), (0, 0, 0))
        padded.paste(image, (pad_width // 2, pad_height // 2))
        image = padded
        width, height = image.size
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def _to_tensor(image: Image.Image) -> np.ndarray:
    """Convert an RGB PIL image to a CHW float32 array scaled to [0, 1]."""
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def _apply_normalize(array: np.ndarray, step: dict) -> np.ndarray:
    """Apply a per-channel 'normalize' transform to a CHW float array."""
    mean = np.array(step.get("mean", 0.0), dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(step.get("std", 1.0), dtype=np.float32).reshape(-1, 1, 1)
    return (array - mean) / std


def preprocess_animetimm_image(image_path: str, preprocess_steps: list) -> np.ndarray:
    """Run the model's declared preprocessing pipeline and return an NCHW batch."""
    data = load_image_rgb(image_path)
    array = None
    for step in preprocess_steps:
        step_type = step.get("type")
        if step_type == "convert_rgb":
            # Already handled by load_image_rgb (opaque RGB on white background).
            continue
        if step_type == "resize":
            data = _apply_resize(data, step)
        elif step_type == "center_crop":
            data = _apply_center_crop(data, step)
        elif step_type in ("to_tensor", "maybe_to_tensor"):
            if array is None:
                array = _to_tensor(data)
        elif step_type == "rescale":
            factor = np.float32(step.get("rescale_factor", 1.0 / 255.0))
            array = (_to_tensor(data) * 255.0 if array is None else array) * factor
        elif step_type == "normalize":
            if array is None:
                array = _to_tensor(data)
            array = _apply_normalize(array, step)
        else:
            log_info(f"animetimm: ignoring unsupported preprocess step '{step_type}'")
    if array is None:
        array = _to_tensor(data)
    return np.expand_dims(array.astype(np.float32), 0)


def load_animetimm_categories(model_path: str) -> dict:
    """Load the category id -> lowercase name mapping from categories.json."""
    with open(os.path.join(model_path, "categories.json"), "r", encoding="utf-8") as f:
        entries = json.load(f)
    category_names = {}
    for entry in entries:
        category_names[int(entry["category"])] = str(entry["name"]).lower()
    return category_names


def load_animetimm_tags(model_path: str) -> list[tuple[str, int]]:
    """Load ordered (tag name, category id) pairs from selected_tags.csv."""
    with open(os.path.join(model_path, "selected_tags.csv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = [h.lower().strip() for h in rows[0]] if rows else []
    try:
        name_idx = header.index("name")
    except ValueError:
        name_idx = 0
    try:
        cat_idx = header.index("category")
    except ValueError:
        cat_idx = 1
    tags = []
    for row in rows[1:]:
        if len(row) <= max(name_idx, cat_idx):
            continue
        try:
            category = int(row[cat_idx])
        except ValueError:
            category = 0
        tags.append((row[name_idx], category))
    return tags


def _threshold_for_category(category_name: str, general_threshold: float, character_threshold: float) -> float:
    """Map an animetimm category name onto SwarmUI's two threshold sliders.

    Rating tags are always excluded (consistent with the other tagger models);
    character tags use the character slider; every other category (general,
    artist, etc.) uses the general slider.
    """
    if category_name == "rating":
        return -1.0
    if category_name == "character":
        return character_threshold
    return general_threshold


def _run_onnx_prediction(model_path: str, img_batch: np.ndarray) -> np.ndarray:
    """Run the model's ONNX export and return the sigmoid prediction vector."""
    import onnxruntime as ort

    providers = onnx_providers()
    log_info(f"Using providers: {providers}")
    sess = ort.InferenceSession(os.path.join(model_path, "model.onnx"), providers=providers)
    input_info = sess.get_inputs()[0]
    output_names = [o.name for o in sess.get_outputs()]
    output_values = sess.run(output_names, {input_info.name: img_batch})
    output_map = {name: val[0] for name, val in zip(output_names, output_values)}
    return output_map.get("prediction", output_values[0][0])


def _ensure_timm() -> None:
    """Ensure the timm package is importable, installing it on demand if needed."""
    try:
        import timm  # noqa: F401
        return
    except ImportError:
        import subprocess
        import sys

        log_info("Installing 'timm' for animetimm safetensors inference...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "timm"], check=True)


def _run_timm_prediction(model_path: str, img_batch: np.ndarray, num_tags: int) -> np.ndarray:
    """Build the timm model from local safetensors weights and return sigmoid probabilities."""
    _ensure_timm()
    import timm
    import torch
    from safetensors.torch import load_file

    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    architecture = config.get("architecture") or config.get("pretrained_cfg", {}).get("architecture")
    if not architecture:
        raise RuntimeError("animetimm config.json is missing the 'architecture' field required to build the timm model.")
    num_classes = config.get("num_classes") or num_tags
    log_info(f"Building timm model '{architecture}' (num_classes={num_classes}) from safetensors weights")

    model = timm.create_model(architecture, pretrained=False, num_classes=num_classes)
    model.load_state_dict(load_file(os.path.join(model_path, "model.safetensors")))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tensor = torch.from_numpy(img_batch).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].float().cpu().numpy()
    return probs


def run_animetimm_inference(image_path: str, repo_id: str, model_dir: str, general_threshold: float, character_threshold: float) -> str:
    """Run an animetimm TIMM tagger (ONNX or safetensors) and return comma-separated tags."""
    log_info(f"Starting animetimm inference: model={repo_id}, general_threshold={general_threshold}, character_threshold={character_threshold}")
    model_path, model_type = ensure_animetimm_model(repo_id, model_dir)

    with open(os.path.join(model_path, "preprocess.json"), "r", encoding="utf-8") as f:
        preprocess_config = json.load(f)
    preprocess_steps = preprocess_config.get("test") or preprocess_config.get("val") or []
    img_batch = preprocess_animetimm_image(image_path, preprocess_steps)
    log_info(f"animetimm input batch shape: {img_batch.shape}, dtype={img_batch.dtype}, model_type={model_type}, min={float(img_batch.min()):.3f}, max={float(img_batch.max()):.3f}")

    category_names = load_animetimm_categories(model_path)
    tags = load_animetimm_tags(model_path)
    log_info(f"Tags loaded: {len(tags)} total across categories {sorted(category_names.items())}")

    if model_type == "onnx":
        probs = _run_onnx_prediction(model_path, img_batch)
    else:
        probs = _run_timm_prediction(model_path, img_batch, len(tags))

    selected = []
    for idx, (tag_name, category) in enumerate(tags):
        if idx >= len(probs):
            break
        category_name = category_names.get(category, "general")
        threshold = _threshold_for_category(category_name, general_threshold, character_threshold)
        if threshold < 0:
            continue
        if float(probs[idx]) >= threshold:
            selected.append(remove_underscore(tag_name))

    log_info(f"Found {len(selected)} animetimm tags above threshold")
    return ", ".join(selected)
