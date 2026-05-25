"""Shared helpers for the SwarmUI WD14Tagger extension."""

import json
import os

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


IMAGE_SIZE = 448


def log_info(msg: str):
    """Emit an info log line to stdout for the C# host to forward."""
    print(json.dumps({"info": msg}), flush=True)


def log_error(msg: str):
    """Emit an error log line to stdout for the C# host to forward."""
    print(json.dumps({"error": msg}), flush=True)


def remove_underscore(tag: str) -> str:
    """Replace underscores with spaces except for very short tags (<=3 chars)."""
    return tag.replace("_", " ") if len(tag) > 3 else tag


def load_image_rgb(image_path: str) -> Image.Image:
    """Load an image and normalize it into an opaque RGB PIL image."""
    image = Image.open(image_path)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def ensure_hf_files(repo_id: str, model_dir: str, filenames: list[str], progress_label: str) -> str:
    """Download a set of HuggingFace files into a repo-specific cache directory."""
    safe_name = repo_id.replace("/", "_")
    model_path = os.path.join(model_dir, safe_name)
    os.makedirs(model_path, exist_ok=True)
    missing = [filename for filename in filenames if not os.path.exists(os.path.join(model_path, filename))]
    if missing:
        print(json.dumps({"progress": progress_label}), flush=True)
        for filename in missing:
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path)
    return model_path


def onnx_providers() -> list[str]:
    """Pick the best available ONNX execution providers."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "ROCMExecutionProvider" in available:
        return ["ROCMExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
