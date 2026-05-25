"""ComfyUI custom node for the SwarmUI WD14Tagger extension.

Runs the extension's existing multi-model tagger inference script inside the ComfyUI
Python environment so expensive tagger execution shares the backend queue with normal
generation work instead of running as a parallel server-side subprocess.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile


_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_deps():
    """Install the WD14Tagger Python requirements into the Comfy Python env if needed."""
    missing = []
    for module_name in ["numpy", "PIL", "huggingface_hub", "onnxruntime"]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if not missing:
        return
    req_path = os.path.join(_EXT_DIR, "requirements.txt")
    if not os.path.exists(req_path):
        raise RuntimeError(f"[WD14Tagger] requirements.txt not found at {req_path}")
    print(f"[WD14Tagger] Installing dependencies for missing modules: {', '.join(missing)}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", req_path], check=True)
    print("[WD14Tagger] Dependencies installed.")


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


class WD14TaggerGenerate:
    """Generate tags from an input image and write them to output_path on disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_id": ("STRING", {"default": "SmilingWolf/wd-eva02-large-tagger-v3"}),
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

    def generate_tags(self, images, model_id, general_threshold, character_threshold, output_path):
        import numpy as np
        from PIL import Image as PILImage

        if not output_path:
            raise ValueError("[WD14Tagger] output_path must not be empty.")

        _ensure_deps()
        _ensure_runtime_modules()

        temp_root = tempfile.mkdtemp(prefix="wd14tagger_")
        temp_image_path = os.path.join(temp_root, "image.png")
        script_path = os.path.join(_EXT_DIR, "wd14_tagger_inference.py")
        model_dir = os.path.abspath(os.path.join(_EXT_DIR, "..", "..", "..", "Models", "wd14_tagger"))

        try:
            image = 255.0 * images[0].cpu().numpy()
            image = PILImage.fromarray(image.clip(0, 255).astype(np.uint8))
            image.save(temp_image_path)

            result = subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--image_path", temp_image_path,
                    "--repo_id", model_id,
                    "--model_dir", model_dir,
                    "--general_threshold", f"{float(general_threshold):.6f}",
                    "--character_threshold", f"{float(character_threshold):.6f}",
                ],
                capture_output=True,
                text=True,
                cwd=_EXT_DIR,
                env=_subprocess_env(),
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

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(tags)
            print(f"[WD14Tagger] Saved tags to {output_path}")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        return ()


NODE_CLASS_MAPPINGS = {
    "WD14TaggerGenerate": WD14TaggerGenerate,
}