"""CLI entry point for the SwarmUI WD14Tagger extension."""

import argparse
import json
import sys

from .camie import CAMIE_MODELS, run_camie_inference
from .pixai import PIXAI_REPO_ID, run_pixai_inference
from .taggerine import TAGGERINE_REPO_ID, run_taggerine_inference
from .wd14 import run_wd14_inference


def run_inference_for_repo(image_path: str, repo_id: str, model_dir: str, general_threshold: float, character_threshold: float) -> str:
    """Dispatch inference to the appropriate backend implementation."""
    if repo_id == TAGGERINE_REPO_ID:
        return run_taggerine_inference(image_path, repo_id, model_dir, general_threshold, character_threshold)
    if repo_id in CAMIE_MODELS:
        return run_camie_inference(image_path, repo_id, model_dir, general_threshold, character_threshold)
    if repo_id == PIXAI_REPO_ID:
        return run_pixai_inference(image_path, repo_id, model_dir, general_threshold, character_threshold)
    return run_wd14_inference(image_path, repo_id, model_dir, general_threshold, character_threshold)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used by the stable top-level script."""
    parser = argparse.ArgumentParser(description="WD14 Tagger inference (SwarmUI extension)")
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
        "--general_threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for general tags (0.0-1.0), or -1.0 to disable.",
    )
    parser.add_argument(
        "--character_threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for character tags (0.0-1.0), or -1.0 to disable.",
    )
    return parser


def main() -> None:
    """Stable CLI wrapper used by the C# API and Comfy node."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        tags = run_inference_for_repo(
            args.image_path,
            args.repo_id,
            args.model_dir,
            args.general_threshold,
            args.character_threshold,
        )
        print(json.dumps({"success": True, "tags": tags}), flush=True)
    except Exception as ex:
        import traceback

        print(json.dumps({"error": traceback.format_exc()}), flush=True)
        print(json.dumps({"success": False, "error": str(ex)}), flush=True)
        sys.exit(1)
