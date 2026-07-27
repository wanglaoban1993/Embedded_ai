import os
import random
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import onnxruntime as ort

# ===== Configuration: modify as needed =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "leaf_model_fp32.onnx")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "data_split")
TEST_DIR = os.path.join(DATA_ROOT, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
INPUT_SIZE = 224

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Occlusion parameters
PATCH_SIZE = 16   # size of the occluding patch
STRIDE = 8        # sliding step


def preprocess_image(image_path: str) -> Tuple[np.ndarray, Image.Image]:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    left = (256 - INPUT_SIZE) // 2
    top = (256 - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

    img_np = np.array(img, dtype=np.float32).transpose(2, 0, 1)
    img_np = (img_np / 255.0 - MEAN) / STD
    img_np = np.expand_dims(img_np, axis=0)
    return img_np.astype(np.float32), img


def pick_random_images(test_dir: str, num_images: int = 5) -> List[str]:
    image_paths: List[str] = []
    for root, _dirs, files in os.walk(test_dir):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(root, name))
    if not image_paths:
        return []
    k = min(num_images, len(image_paths))
    return random.sample(image_paths, k)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def occlusion_map(session: ort.InferenceSession, input_tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_tensor})[0]
    probs = _softmax(logits)
    target_idx = int(np.argmax(probs, axis=1)[0])
    base_score = float(probs[0, target_idx])

    _, _, h, w = input_tensor.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            occluded = input_tensor.copy()
            occluded[:, :, y:y + PATCH_SIZE, x:x + PATCH_SIZE] = 0.0
            o_logits = session.run(None, {input_name: occluded})[0]
            o_probs = _softmax(o_logits)
            o_score = float(o_probs[0, target_idx])

            drop = max(base_score - o_score, 0.0)
            heatmap[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += drop
            counts[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1.0

    counts[counts == 0] = 1.0
    heatmap = heatmap / counts
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def save_overlay(image: Image.Image, heatmap: np.ndarray, out_name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    heat = (heatmap * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(image.size, resample=Image.BILINEAR)
    heat_img = heat_img.convert("RGBA")

    r, g, b, _ = heat_img.split()
    heat_rgba = Image.merge("RGBA", (r, Image.new("L", r.size, 0), Image.new("L", r.size, 0), r))

    base = image.convert("RGBA")
    overlay = Image.alpha_composite(base, heat_rgba)
    overlay.save(os.path.join(OUTPUT_DIR, f"{out_name}_overlay.png"))
    heat_img.save(os.path.join(OUTPUT_DIR, f"{out_name}_heatmap.png"))
    image.save(os.path.join(OUTPUT_DIR, f"{out_name}_original.png"))


def main() -> None:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test dir not found: {TEST_DIR}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

    image_paths = pick_random_images(TEST_DIR, num_images=5)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {TEST_DIR}")

    for idx, image_path in enumerate(image_paths, start=1):
        input_tensor, image = preprocess_image(image_path)
        heatmap = occlusion_map(session, input_tensor)
        save_overlay(image, heatmap, f"onnx_occlusion_{idx}")

    print(f"Saved occlusion XAI to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()