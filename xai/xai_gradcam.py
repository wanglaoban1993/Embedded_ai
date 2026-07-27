import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

# ===== Configuration: modify as needed =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))
from test_onnx_fp32 import pick_test_images

WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "mobilenet_v3_small_7cls.pth")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "data_split")
TEST_DIR = os.path.join(DATA_ROOT, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
NUM_CLASSES = 7
INPUT_SIZE = 224

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    left = (256 - INPUT_SIZE) // 2
    top = (256 - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - MEAN) / STD
    img_np = img_np.transpose(2, 0, 1)
    tensor = torch.from_numpy(img_np).unsqueeze(0)
    return tensor


def pick_images_from_test(num_images: int = 5) -> list:
    return pick_test_images(num_images=num_images)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _input, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward()

        grads = self.gradients
        acts = self.activations
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1).squeeze(0)
        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        cam = cam.cpu().numpy()
        return cam


def save_cam_overlay(image_path: str, cam: np.ndarray, out_prefix: str) -> None:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE))

    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (INPUT_SIZE, INPUT_SIZE), resample=Image.BILINEAR
    )
    cam_rgb = Image.merge("RGB", (cam_resized, cam_resized, cam_resized))

    overlay = Image.blend(img, cam_rgb, alpha=0.5)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overlay.save(os.path.join(OUTPUT_DIR, f"{out_prefix}_overlay.png"))
    cam_resized.save(os.path.join(OUTPUT_DIR, f"{out_prefix}_heatmap.png"))


def main() -> None:
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

    image_paths = pick_images_from_test(num_images=5)
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {TEST_DIR}")

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # MobileNetV3 Small: use last conv block in features
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)

    for image_path in image_paths:
        input_tensor = preprocess_image(image_path)
        heatmap = cam.generate(input_tensor)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_cam_overlay(image_path, heatmap, base_name)

        print(f"Image: {image_path}")
        print(f"Saved Grad-CAM to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
