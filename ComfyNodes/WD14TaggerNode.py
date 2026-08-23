"""ComfyUI custom node for the SwarmUI WD14Tagger extension.

Runs the extension's existing multi-model tagger inference script inside the ComfyUI
Python environment so expensive tagger execution shares the backend queue with normal
generation work instead of running as a parallel server-side subprocess.
"""

import json
import math
import os
import re
import shutil
# A subprocess keeps tagger allocations isolated and is invoked without a shell.
import subprocess  # nosec B404
import sys
import tempfile

_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

from wd14tagger.models import require_supported_model  # noqa: E402

_OUTPUT_NAME_PATTERN = re.compile(r"^wd14tagger_[0-9a-f]{32}\.txt$")
_SUBPROCESS_TIMEOUT_SECONDS = 60 * 60


def _require_deps():
    """Require extension dependencies without modifying the Comfy environment at runtime."""
    missing = []
    for module_name in ["numpy", "PIL", "huggingface_hub", "onnxruntime"]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if not missing:
        return
    req_path = os.path.join(_EXT_DIR, "requirements.txt")
    raise RuntimeError(
        "[WD14Tagger] Missing Python modules: "
        f"{', '.join(missing)}. Install the reviewed dependencies with "
        f"'{sys.executable} -m pip install -r {req_path}', then restart SwarmUI. "
        "The extension does not install packages during a generation."
    )


def _ensure_runtime_modules():
    """Verify modules expected from the existing Comfy runtime are available."""
    missing = []
    for module_name in ["requests", "safetensors", "torch", "torchvision"]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "[WD14Tagger] Missing Comfy runtime modules: "
            f"{', '.join(missing)}. This extension expects those to come from the existing Comfy Python runtime rather than installing them itself."
        )


def _parse_result(stdout: str) -> str:
    """Read the final success JSON line from the inference script output."""
    result = None
    for line in stdout.splitlines():
        trimmed = line.strip()
        if not trimmed.startswith("{"):
            continue
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            continue
        if "info" in parsed:
            print(f"[WD14Tagger] {parsed['info']}")
        elif "progress" in parsed:
            print(f"[WD14Tagger] {parsed['progress']}")
        elif "error" in parsed:
            print(f"[WD14Tagger] Error: {parsed['error']}")
        if "success" in parsed:
            result = parsed
    if result is None:
        raise RuntimeError("Tagger produced no final JSON result.")
    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Unknown tagger error.")
    return result.get("tags", "")


def _subprocess_env() -> dict:
    """Build a subprocess environment that can import the extension package reliably."""
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _EXT_DIR if not current else os.pathsep.join([_EXT_DIR, current])
    return env


def _validated_output_path(output_path: str) -> str:
    """Only allow the random temporary output files created by the Swarm API."""
    resolved = os.path.normcase(os.path.realpath(os.path.abspath(output_path or "")))
    allowed_temp_dirs = {os.path.normcase(os.path.realpath(tempfile.gettempdir()))}
    if local_app_data := os.environ.get("LOCALAPPDATA"):
        allowed_temp_dirs.add(os.path.normcase(os.path.realpath(os.path.join(local_app_data, "Temp"))))
    if os.path.dirname(resolved) not in allowed_temp_dirs or not _OUTPUT_NAME_PATTERN.fullmatch(os.path.basename(resolved)):
        raise ValueError("[WD14Tagger] Refusing an output path outside the WD14Tagger temporary-file namespace.")
    return resolved


def _validated_model_directory(model_directory: str, model_id: str) -> str:
    """Require the model path shape produced by SwarmUI's configured model roots."""
    default_model_dir = os.path.join(
        _EXT_DIR, "..", "..", "..", "Models", "wd14_tagger", model_id.replace("/", "_")
    )
    resolved = os.path.realpath(os.path.abspath(os.path.expanduser((model_directory or "").strip() or default_model_dir)))
    expected_leaf = model_id.replace("/", "_")
    if os.path.basename(resolved) != expected_leaf or os.path.basename(os.path.dirname(resolved)).lower() != "wd14_tagger":
        raise ValueError("[WD14Tagger] Refusing a model directory outside the expected wd14_tagger/model-id layout.")
    return resolved


def _validated_threshold(value, name: str) -> float:
    """Validate thresholds before forwarding them to the inference process."""
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed != -1.0 and not 0.0 <= parsed <= 1.0):
        raise ValueError(f"[WD14Tagger] {name} must be between 0.0 and 1.0, or -1.0 to disable it.")
    return parsed


class WD14TaggerGenerate:
    """Generate tags from an input image and write them to output_path on disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_id": ("STRING", {"default": "SmilingWolf/wd-eva02-large-tagger-v3"}),
                "model_directory": ("STRING", {"default": ""}),
                "general_threshold": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01}),
                "character_threshold": ("FLOAT", {"default": 0.85, "min": -1.0, "max": 1.0, "step": 0.01}),
                "output_path": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "WD14Tagger"
    RETURN_TYPES = ()
    FUNCTION = "generate_tags"
    OUTPUT_NODE = True
    DESCRIPTION = "Runs the SwarmUI WD14 tagger through the Comfy backend queue and writes the tags to output_path."

    def generate_tags(self, images, model_id, model_directory, general_threshold, character_threshold, output_path):
        import numpy as np
        from PIL import Image as PILImage

        model_id = require_supported_model(model_id)
        output_path = _validated_output_path(output_path)
        model_dir = _validated_model_directory(model_directory, model_id)
        general_threshold = _validated_threshold(general_threshold, "general_threshold")
        character_threshold = _validated_threshold(character_threshold, "character_threshold")

        _require_deps()
        _ensure_runtime_modules()

        temp_root = tempfile.mkdtemp(prefix="wd14tagger_")
        temp_image_path = os.path.join(temp_root, "image.png")
        script_path = os.path.join(_EXT_DIR, "wd14_tagger_inference.py")
        try:
            image = 255.0 * images[0].cpu().numpy()
            image = PILImage.fromarray(image.clip(0, 255).astype(np.uint8))
            image.save(temp_image_path)

            # The executable and script are fixed; every forwarded input is validated above.
            result = subprocess.run(  # nosec B603
                [
                    sys.executable,
                    script_path,
                    "--image_path", temp_image_path,
                    "--repo_id", model_id,
                    "--model_dir", model_dir,
                    "--general_threshold", f"{general_threshold:.6f}",
                    "--character_threshold", f"{character_threshold:.6f}",
                ],
                capture_output=True,
                text=True,
                cwd=_EXT_DIR,
                env=_subprocess_env(),
                timeout=_SUBPROCESS_TIMEOUT_SECONDS,
                check=False,
            )
            if result.stderr.strip():
                print(f"[WD14Tagger] stderr: {result.stderr.strip()}")
            if result.returncode != 0:
                raise RuntimeError(
                    "Tagger subprocess failed"
                    f" (exit {result.returncode})."
                    f" stdout: {result.stdout.strip() or '<empty>'}"
                    f" stderr: {result.stderr.strip() or '<empty>'}"
                )
            tags = _parse_result(result.stdout)

            with open(output_path, "x", encoding="utf-8") as f:
                f.write(tags)
            print(f"[WD14Tagger] Saved tags to {output_path}")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        return ()


NODE_CLASS_MAPPINGS = {
    "WD14TaggerGenerate": WD14TaggerGenerate,
}
