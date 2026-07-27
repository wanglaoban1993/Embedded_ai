import json
import os
import random
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

try:
    from torchvision import datasets
except Exception:
    datasets = None

# ===== setting up =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "leaf_model_fp32.onnx")
IMAGE_PATH = os.path.join(BASE_DIR, "test.jpg")
LABELS_JSON = os.path.join(PROJECT_ROOT, "data", "class_indices.json")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "data_split")
TEST_DIR = os.path.join(DATA_ROOT, "test")
INPUT_SIZE = 224

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_labels() -> list:
    if os.path.isfile(LABELS_JSON):
        with open(LABELS_JSON, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # case A: {"0": "cat", "1": "dog"}
        if all(str(k).isdigit() for k in mapping.keys()):
            ordered = sorted(((int(k), v) for k, v in mapping.items()), key=lambda x: x[0])
            return [label for _, label in ordered]

        # case B: {"cat": 0, "dog": 1}
        if all(isinstance(v, int) or (isinstance(v, str) and str(v).isdigit()) for v in mapping.values()):
            inv = {int(v): k for k, v in mapping.items()}
            return [inv[i] for i in range(len(inv))]

    val_dir = os.path.join(DATA_ROOT, "val")
    if datasets is not None and os.path.isdir(val_dir):
        dataset = datasets.ImageFolder(val_dir)
        return list(dataset.classes)

    return []


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    left = (256 - INPUT_SIZE) // 2
    top = (256 - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

    img_data = np.array(img, dtype=np.float32)
    img_data = img_data.transpose(2, 0, 1)
    img_data = (img_data / 255.0 - MEAN) / STD
    img_data = img_data.astype(np.float32)
    img_data = np.expand_dims(img_data, axis=0)
    return img_data


def pick_test_images(num_images: int = 2) -> list:
    if not os.path.isdir(TEST_DIR):
        return []

    class_dirs = [
        os.path.join(TEST_DIR, d)
        for d in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, d))
    ]
    if not class_dirs:
        return []

    selected = []
    random.shuffle(class_dirs)
    for class_dir in class_dirs:
        images = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        if images:
            selected.append(random.choice(images))
            if len(selected) >= num_images:
                return selected

    # fallback: sample from all images if not enough classes
    all_images = []
    for root, _, files in os.walk(TEST_DIR):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                all_images.append(os.path.join(root, name))

    remaining = num_images - len(selected)
    if remaining > 0 and all_images:
        pool = [p for p in all_images if p not in selected]
        selected.extend(random.sample(pool, k=min(remaining, len(pool))))

    return selected


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def main() -> None:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    image_paths = []
    if os.path.isfile(IMAGE_PATH):
        image_paths = [IMAGE_PATH]
    else:
        image_paths = pick_test_images(num_images=5)
        if not image_paths:
            raise FileNotFoundError(
                f"Image not found: {IMAGE_PATH} and no images in {TEST_DIR}"
            )

    labels = load_labels()
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for image_path in image_paths:
        input_tensor = preprocess_image(image_path)
        outputs = session.run(None, {input_name: input_tensor})

        logits = outputs[0]
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        score = float(probs[0][idx])

        label = labels[idx] if labels and idx < len(labels) else str(idx)
        print(f"Image: {image_path}")
        print(f"Top-1: {label} (score={score:.4f})")


if __name__ == "__main__":
    main()
