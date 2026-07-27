# export_model_to_onnx.py

import torch
import torch.nn as nn
import os

# --- 0. Define the model structure (must match training) ---
# Assumes we have the MLP classifier trained in a previous chapter
class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # input_dim assumed to be 784
        x = x.view(-1, input_dim)
        out = self.fc2(self.relu(self.fc1(x)))
        return out

# --- 1. Load the trained PyTorch model ---
print("--- Case #001: Export a PyTorch model to ONNX ---")
INPUT_DIM = 28 * 28
HIDDEN_DIM = 256
OUTPUT_DIM = 10
MODEL_PATH = 'mnist_results/simple_mnist_mlp.pth' # assumed path to the model weights saved earlier

model = SimpleMLPClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    print(f"[OK] PyTorch model weights successfully loaded from '{MODEL_PATH}'.")
except FileNotFoundError:
    print(f"[ERROR] Model weights file '{MODEL_PATH}' not found. Please make sure the file exists!")
    exit()
model.eval() # set to evaluation mode

# --- 2. Prepare a dummy input ---
# ONNX export needs a sample input to trace the model's computation graph
dummy_input = torch.randn(1, 1, 28, 28) # a single MNIST image [Batch=1, C=1, H=28, W=28]

# --- 3. Export the model to ONNX format ---
onnx_path = 'simple_mnist_mlp.onnx'
print(f"\nExporting model to ONNX format: {onnx_path}...")

# torch.onnx.export() is PyTorch's core function for exporting to ONNX
torch.onnx.export(
    model,                      # the PyTorch model to export
    dummy_input,                # sample input for the model, used to trace the computation graph
    onnx_path,                  # path to the exported file
    opset_version=11,           # ONNX opset version, usually set to 11 or higher
    input_names=['input'],      # name of the input node
    output_names=['output'],    # name of the output node
    dynamic_axes={              # define dynamic axes, i.e. dimensions that can vary at runtime (e.g. batch size)
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print(f"[OK] Model successfully exported to ONNX format!")
