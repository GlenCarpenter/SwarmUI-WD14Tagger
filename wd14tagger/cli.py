"""CLI entry point for the SwarmUI WD14Tagger extension."""

import argparse
import json
import math
import os
import sys

from .animetimm import is_animetimm_repo, run_animetimm_inference
from .camie import CAMIE_MODELS, run_camie_inference
from .common import TaggerUserError
from .joytag import JOYTAG_REPO_ID, run_joytag_inference
from .models import require_supported_model
from .pixai import PIXAI_REPO_ID, run_pixai_inference
from .taggerine import TAGGERINE_REPO_ID, run_taggerine_inference
from .wd14 import run_wd14_inference


def run_inference_for_repo(image_path: str, repo_id: str, model_dir: str, general_threshold: float, character_threshold: float) -> str:
    """Dispatch inference to the appropriate backend implementation."""
    if repo_id == JOYTAG_REPO_ID:
        return run_joytag_inference(image_path, repo_id, model_dir, general_threshold, character_threshold)
    if repo_id == TAGGERINE_REPO_ID:
        return run_taggerine_inference(image_path, repo_id, model_dir, general_threshold, character_threshold)
    if repo_id in CAMIE_MODELS:
        return run_camie_inference(image_path, repo_id, model_dir, general_threshold, character_threshold)
    if repo_id == PIXAI_REPO_ID:
        return run_pixai_inference(image_path, repo_id, model_dir, general_threshold, character_threshold)
    if is_animetimm_repo(repo_id):
        return run_animetimm_inference(image_path, repo_id, model_dir, general_threshold, character_threshold)
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
        default="",
        help="Local directory containing the selected model's files",
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


def _validate_threshold(value: float, name: str) -> float:
    """Reject non-finite and out-of-range confidence thresholds."""
    if not math.isfinite(value) or (value != -1.0 and not 0.0 <= value <= 1.0):
        raise TaggerUserError(f"{name} must be between 0.0 and 1.0, or -1.0 to disable it.")
    return value


def main() -> None:
    """Stable CLI wrapper used by the C# API and Comfy node."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        try:
            repo_id = require_supported_model(args.repo_id)
        except ValueError as ex:
            raise TaggerUserError(str(ex)) from ex
        general_threshold = _validate_threshold(args.general_threshold, "General threshold")
        character_threshold = _validate_threshold(args.character_threshold, "Character threshold")
        model_dir = args.model_dir or os.path.join("Models", "wd14_tagger", repo_id.replace("/", "_"))
        tags = run_inference_for_repo(
            args.image_path,
            repo_id,
            model_dir,
            general_threshold,
            character_threshold,
        )
        print(json.dumps({"success": True, "tags": tags}), flush=True)
    except TaggerUserError as ex:
        # Clean, user-facing failures: surface just the actionable message, no traceback wall.
        print(json.dumps({"success": False, "error": str(ex)}), flush=True)
        sys.exit(1)
    except Exception as ex:
        import traceback

        print(json.dumps({"error": traceback.format_exc()}), flush=True)
        print(json.dumps({"success": False, "error": str(ex)}), flush=True)
        sys.exit(1)
