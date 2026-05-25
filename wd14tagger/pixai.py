"""PixAI ONNX inference helpers."""

import numpy as np
from PIL import Image

from .common import IMAGE_SIZE, load_image_rgb, log_info, onnx_providers, remove_underscore
from .wd14 import ensure_wd14_model, load_wd14_tags


PIXAI_REPO_ID = "deepghs/pixai-tagger-v0.9-onnx"


def preprocess_pixai_image(image_path: str) -> np.ndarray:
    """Preprocess an image for PixAI NCHW/RGB ONNX models."""
    image = load_image_rgb(image_path)
    img = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)


def run_pixai_inference(image_path: str, repo_id: str, model_dir: str, general_threshold: float, character_threshold: float) -> str:
    """Run PixAI ONNX inference and return comma-separated tags."""
    import onnxruntime as ort

    log_info(f"Starting PixAI inference: model={repo_id}, general_threshold={general_threshold}, character_threshold={character_threshold}")
    model_path = ensure_wd14_model(repo_id, model_dir)
    providers = onnx_providers()
    log_info(f"Using providers: {providers}")

    sess = ort.InferenceSession(f"{model_path}/model.onnx", providers=providers)
    input_info = sess.get_inputs()[0]
    img_batch = preprocess_pixai_image(image_path)
    log_info(f"PixAI input batch shape: {img_batch.shape}, dtype={img_batch.dtype}, min={float(img_batch.min()):.3f}, max={float(img_batch.max()):.3f}")

    output_names = [o.name for o in sess.get_outputs()]
    output_values = sess.run(output_names, {input_info.name: img_batch})
    output_map = {name: val[0] for name, val in zip(output_names, output_values)}
    probs = output_map.get("prediction", output_values[0][0])

    general_tags, character_tags, rating_count = load_wd14_tags(model_path)
    tags = []
    for i, prob in enumerate(probs[rating_count:]):
        if i < len(general_tags):
            if general_threshold >= 0 and prob >= general_threshold:
                tags.append(remove_underscore(general_tags[i]))
        else:
            char_idx = i - len(general_tags)
            if char_idx < len(character_tags) and character_threshold >= 0 and prob >= character_threshold:
                tags.append(remove_underscore(character_tags[char_idx]))

    log_info(f"Found {len(tags)} PixAI tags above threshold")
    return ", ".join(tags)
