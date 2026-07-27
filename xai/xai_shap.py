import os
import random
import numpy as np
from PIL import Image
import onnxruntime as ort
import shap
import matplotlib.pyplot as plt

# ===== Configuration =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "leaf_model_fp32.onnx")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "data_split")
TEST_DIR = os.path.join(DATA_ROOT, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
INPUT_SIZE = 224

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    left = (256 - INPUT_SIZE) // 2
    top = (256 - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    img_np = np.array(img, dtype=np.float32) / 255.0
    return img_np  # HWC, float32 in [0,1]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def predict_fn(images: np.ndarray) -> np.ndarray:
    # images: (N, H, W, C), float32 [0,1]

    # x = (images - MEAN) / STD
    # x = x.transpose(0, 3, 1, 2).astype(np.float32)  # NCHW
    # logits = session.run(None, {input_name: x})[0]
    # probs = softmax(logits)
    # return probs    
    outputs = []
    for i in range(images.shape[0]):
        img = images[i : i + 1]  # force batch=1
        x = (img - MEAN) / STD
        x = x.transpose(0, 3, 1, 2).astype(np.float32)  # NCHW
        logits = session.run(None, {input_name: x})[0]
        probs = softmax(logits)
        outputs.append(probs)
    return np.concatenate(outputs, axis=0)


def pick_random_images(test_dir: str, num_images: int = 5) -> list:
    image_paths = []
    for root, _dirs, files in os.walk(test_dir):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(root, name))
    if not image_paths:
        return []
    k = min(num_images, len(image_paths))
    return random.sample(image_paths, k)


def main() -> None:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test dir not found: {TEST_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    global session, input_name
    session = ort.InferenceSession(
        MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    image_paths = pick_random_images(TEST_DIR, num_images=5)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {TEST_DIR}")

    for idx, image_path in enumerate(image_paths, start=1):
        image = preprocess_image(image_path)
        image_batch = np.expand_dims(image, axis=0)

        masker = shap.maskers.Image("blur(8,8)", image.shape)
        explainer = shap.Explainer(predict_fn, masker)
        shap_values = explainer(image_batch, max_evals=50, batch_size=1)

        plt.figure()
        shap.image_plot(shap_values, image_batch, show=False)
        out_path = os.path.join(OUTPUT_DIR, f"shap_overlay_{idx}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved SHAP to: {out_path}")


if __name__ == "__main__":
    main()