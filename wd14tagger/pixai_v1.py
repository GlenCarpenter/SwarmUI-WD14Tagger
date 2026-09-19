"""PixAI Tagger v1.0 Transformers inference helpers."""

from .common import ensure_hf_files, load_image_rgb, log_info, remove_underscore


PIXAI_V1_REPO_ID = "pixai-labs/pixai-tagger-v1.0"
PIXAI_V1_FILES = ["config.json", "preprocessor_config.json", "model.safetensors", "tagger_pipeline.py"]


def ensure_pixai_v1_model(repo_id: str, model_dir: str) -> str:
    """Download the PixAI v1 model and custom Transformers pipeline when missing."""
    return ensure_hf_files(repo_id, model_dir, PIXAI_V1_FILES, f"Downloading PixAI Tagger v1.0 model {repo_id}...")


def _threshold_for_category(category: str, general_threshold: float, character_threshold: float) -> float:
    """Map PixAI v1 categories onto SwarmUI's two threshold controls."""
    if category == "rating":
        return 1.0
    if category == "character":
        return character_threshold if character_threshold >= 0 else 1.0
    return general_threshold if general_threshold >= 0 else 1.0


def run_pixai_v1_inference(image_path: str, repo_id: str, model_dir: str, general_threshold: float, character_threshold: float) -> str:
    """Run PixAI Tagger v1.0 inference and return comma-separated tags."""
    import torch
    from transformers import pipeline

    log_info(f"Starting PixAI v1 inference: model={repo_id}, general_threshold={general_threshold}, character_threshold={character_threshold}")
    model_path = ensure_pixai_v1_model(repo_id, model_dir)
    device = 0 if torch.cuda.is_available() else -1
    log_info(f"PixAI v1 device: {'cuda' if device == 0 else 'cpu'}")

    tagger = pipeline(model=model_path, image_processor=model_path, trust_remote_code=True, device=device)
    thresholds = {
        str(category): _threshold_for_category(str(category), general_threshold, character_threshold)
        for category, _ in tagger.model.config.tags_split
    }
    results = tagger(load_image_rgb(image_path), threshold=thresholds)["results"]

    tags = []
    for category, category_tags in results.items():
        if category == "rating":
            continue
        for tag_name in category_tags:
            tags.append(remove_underscore(tag_name))

    log_info(f"Found {len(tags)} PixAI v1 tags above threshold")
    return ", ".join(tags)