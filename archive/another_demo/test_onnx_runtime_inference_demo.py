# onnx_runtime_inference_demo.py

import onnxruntime as ort # import the ONNX Runtime library
import numpy as np
import torch # used to generate a dummy input and for comparison
from PIL import Image
import torchvision.transforms as T # used for image preprocessing
import os

# --- 1. Prepare the ONNX model path (assumed already exported) ---
ONNX_MODEL_PATH = 'simple_mnist_mlp.onnx' # ONNX model exported in the previous section

if not os.path.exists(ONNX_MODEL_PATH):
    print(f"[ERROR] ONNX model file '{ONNX_MODEL_PATH}' not found. Please run the export step first!")
    exit()

# --- 2. Case #001: Verify inference using ONNX Runtime ---
print("--- Case #001: Verify inference using ONNX Runtime ---")

# 2.1 Create an ONNX Runtime session
# The providers argument selects the backend, e.g. 'CUDAExecutionProvider' (GPU) or 'CPUExecutionProvider' (CPU)
# By default, ONNX Runtime tries to use the best available provider
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider']) # force CPU for this demo

# 2.2 Get the model's input/output node names (must match what was defined at export time)
input_name = session.get_inputs()[0].name # get the model's input node name, i.e. 'input'
output_name = session.get_outputs()[0].name # get the model's output node name, i.e. 'output'
print(f"ONNX model input name: '{input_name}', output name: '{output_name}'")

# 2.3 Prepare test data (same as PyTorch)
# Assume we have a PyTorch Tensor of an MNIST image
# We need to convert it to a NumPy array with the shape expected by the ONNX model [Batch, C, H, W]
# Here we use random data to simulate an image
dummy_input_pytorch = torch.randn(1, 1, 28, 28)
input_data_np = dummy_input_pytorch.numpy().astype(np.float32) # convert to a NumPy array and ensure the dtype

print(f"Prepared inference input data shape (NumPy): {input_data_np.shape}")

# 2.4 Run inference
# session.run() takes a list of output names and an input dictionary
# The input dictionary's keys are the model's input names (input_name), values are NumPy arrays
output_onnx = session.run([output_name], {input_name: input_data_np})
predicted_logits_np = output_onnx[0] # ONNX Runtime returns a list of results, take the first one

print(f"ONNX Runtime inference output shape (NumPy): {predicted_logits_np.shape}")

# 2.5 Post-process the results (same as PyTorch)
# Get the predicted class
predicted_class_id = np.argmax(predicted_logits_np, axis=1)[0]
print(f"ONNX Runtime predicted class ID: {predicted_class_id}")

print("\n[OK] ONNX Runtime inference verification complete!")
print("This simulates the core process of loading an ONNX model on an edge device for inference.")
