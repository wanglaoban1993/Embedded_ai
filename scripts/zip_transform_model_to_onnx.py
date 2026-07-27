import os

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ===== Configuration: modify according to your training results =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "mobilenet_v3_small_7cls.pth")
NUM_CLASSES = 7
INPUT_SIZE = 224
ONNX_OUT = os.path.join(PROJECT_ROOT, "models", "leaf_model_fp32.onnx")
EXPORT_INT8 = False
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "data_split")
CALIBRATION_DIR = os.path.join(DATA_ROOT, "val")
CALIBRATION_SAMPLES = 200
CALIBRATION_BATCH_SIZE = 32

# 1) build the same model as training time
model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

# 2) load the trained weights
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# 3) export FP32 ONNX (make sure it runs correctly first)
dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_OUT,
    opset_version=13,
    input_names=["input"],
    output_names=["logits"],
)

# ===== Optional: if you want to do PyTorch static quantization (better for ARM's qnnpack) =====
# Note: It's best to perform quantization on a PC, then copy to Pi Zero WH.
# If you don't do quantization, keeping FP32 can still run on ONNX Runtime, just slower.
#
if EXPORT_INT8:
    if os.name == "nt":
        raise RuntimeError(
            "INT8 quantization export on Windows may trigger missing quantized operators."
            "Please perform quantization export on Linux/macOS, or export FP32 first."
        )

    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    model_fp32_prepared = torch.quantization.prepare(model)

# Use the validation set for calibration (only take a small sample)
    if os.path.isdir(CALIBRATION_DIR):
        val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        calibration_dataset = datasets.ImageFolder(CALIBRATION_DIR, transform=val_transform)
        total = len(calibration_dataset)
        sample_count = min(CALIBRATION_SAMPLES, total)
        if sample_count > 0:
            indices = torch.randperm(total)[:sample_count].tolist()
            calibration_subset = Subset(calibration_dataset, indices)
            calibration_loader = DataLoader(
                calibration_subset,
                batch_size=CALIBRATION_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            model_fp32_prepared.eval()
            with torch.no_grad():
                for images, _ in calibration_loader:
                    model_fp32_prepared(images)
    else:
        print(f"Calibration dir not found: {CALIBRATION_DIR}. Skip calibration.")

    model_int8 = torch.quantization.convert(model_fp32_prepared)
    torch.onnx.export(
        model_int8,
        dummy_input,
        os.path.join(PROJECT_ROOT, "models", "leaf_model_int8.onnx"),
        opset_version=13,
        input_names=["input"],
        output_names=["logits"],
    )