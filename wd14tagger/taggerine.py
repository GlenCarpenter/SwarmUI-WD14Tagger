"""Taggerine PyTorch inference helpers."""

import importlib.util
import json
import os
import sys

from .common import ensure_hf_files, log_info, remove_underscore


TAGGERINE_REPO_ID = "lodestones/taggerine"
TAGGERINE_CHECKPOINT = "tagger_proto.safetensors"
TAGGERINE_VOCAB = "tagger_vocab_with_categories_and_alias_updated.json"
TAGGERINE_SCRIPT = "inference_tagger_standalone.py"


def ensure_taggerine_assets(repo_id: str, model_dir: str) -> tuple[str, str, str]:
    """Download Taggerine assets when missing."""
    model_path = ensure_hf_files(
        repo_id,
        model_dir,
        [TAGGERINE_CHECKPOINT, TAGGERINE_VOCAB, TAGGERINE_SCRIPT],
        f"Downloading Taggerine assets for {repo_id}...",
    )
    return (
        os.path.join(model_path, TAGGERINE_CHECKPOINT),
        os.path.join(model_path, TAGGERINE_VOCAB),
        os.path.join(model_path, TAGGERINE_SCRIPT),
    )


def load_taggerine_vocab(vocab_path: str) -> dict:
    """Load Taggerine vocab metadata and normalize category lookup."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tag2category", {})


def load_taggerine_module(script_path: str):
    """Import the upstream Taggerine standalone script as a local module."""
    module_name = f"taggerine_inference_{abs(hash(script_path))}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Taggerine inference module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def taggerine_threshold_for_category(category, general_threshold: float, character_threshold: float) -> float:
    """Map Taggerine categories onto SwarmUI's existing threshold UI."""
    if isinstance(category, str):
        category_lower = category.lower()
        if category_lower == "character":
            return character_threshold
        if category_lower == "rating":
            return -1.0
        return general_threshold
    try:
        category_id = int(category)
    except (TypeError, ValueError):
        return general_threshold
    if category_id == 4:
        return character_threshold
    if category_id == 9:
        return -1.0
    return general_threshold


def pick_taggerine_base_threshold(general_threshold: float, character_threshold: float) -> float:
    """Return the lowest enabled threshold so upstream inference returns all locally relevant tags."""
    enabled = [threshold for threshold in (general_threshold, character_threshold) if threshold >= 0]
    return min(enabled) if enabled else -1.0


def run_taggerine_inference(image_path: str, repo_id: str, model_dir: str, general_threshold: float, character_threshold: float) -> str:
    """Run Taggerine inference and return comma-separated tags."""
    base_threshold = pick_taggerine_base_threshold(general_threshold, character_threshold)
    if base_threshold < 0:
        log_info("Taggerine thresholds are both disabled; returning no tags")
        return ""

    log_info(f"Starting Taggerine inference: model={repo_id}, general_threshold={general_threshold}, character_threshold={character_threshold}")
    checkpoint_path, vocab_path, script_path = ensure_taggerine_assets(repo_id, model_dir)
    taggerine_module = load_taggerine_module(script_path)
    category_map = load_taggerine_vocab(vocab_path)

    device = "cuda" if taggerine_module.torch.cuda.is_available() else "cpu"
    dtype = taggerine_module.torch.float32
    if device == "cuda":
        capability = taggerine_module.torch.cuda.get_device_capability()
        dtype = taggerine_module.torch.bfloat16 if capability[0] >= 8 else taggerine_module.torch.float16
    log_info(f"Taggerine device: {device}, dtype={dtype}")

    tagger = taggerine_module.Tagger(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        device=device,
        dtype=dtype,
    )
    results = tagger.predict(image_path, topk=None, threshold=base_threshold)

    tags = []
    for tag_name, score in results:
        threshold = taggerine_threshold_for_category(category_map.get(tag_name), general_threshold, character_threshold)
        if threshold < 0 or score < threshold:
            continue
        tags.append(remove_underscore(tag_name))

    log_info(f"Found {len(tags)} Taggerine tags above threshold")
    return ", ".join(tags)
