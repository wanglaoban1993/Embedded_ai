# LRP and SHAP for explainability are impossible to
# implement on the embedded device due to resource constraints.

# File name: export_xai_model.py
# Runtime environment: PC / Google Colab (PyTorch must be installed)

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# ================= 1. Define an "explainable" wrapper class =================
class ExplainableMobileNet(nn.Module):
    def __init__(self, num_classes=38): # assumes you have 38 classes
        super(ExplainableMobileNet, self).__init__()
        # Load the pretrained model (assumes you've already fine-tuned it, or will load weights)
        # If you have a trained .pth file, load the original structure first, then load_state_dict
        original_model = models.mobilenet_v3_small(weights=None)

        # Modify the classifier head to match your number of classes (must match training)
        in_features = original_model.classifier[3].in_features
        original_model.classifier[3] = nn.Linear(in_features, num_classes)

        # Load your trained weights (very important! otherwise the output is random)
        # original_model.load_state_dict(torch.load("your_trained_weights.pth"))

        # Disassemble the model: extract the feature layers and the classifier layers
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.flatten = nn.Flatten()
        self.classifier = original_model.classifier

    def forward(self, x):
        # 1. Extract the feature maps (this is the core data CAM needs)
        # MobileNetV3 Small's feature map output is usually (Batch, 576, 7, 7)
        feature_maps = self.features(x)

        # 2. Continue the normal classification flow
        x = self.avgpool(feature_maps)
        x = self.flatten(x)
        logits = self.classifier(x)

        # 3. Return both: classification scores + feature maps
        return logits, feature_maps

# ================= 2. Export the model =================
def main():
    print("Preparing model...")
    # Instantiate the model
    num_classes = 38 # change to your actual number of classes
    model = ExplainableMobileNet(num_classes=num_classes)
    model.eval()

    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    print("Exporting model_xai.onnx ...")
    torch.onnx.export(
        model,
        dummy_input,
        "model_xai.onnx",
        opset_version=12,
        input_names=['input_image'],
        output_names=['class_logits', 'feature_maps'] # these names matter, used on the Raspberry Pi
    )

    # ================= 3. Extract and save the classifier weights =================
    # CAM also needs the weights of the final fully-connected layer
    # MobileNetV3's classifier structure:
    # classifier[0]: Linear
    # classifier[1]: Hardswish
    # classifier[2]: Dropout
    # classifier[3]: Linear (this is the one we want, responsible for the final classification)

    print("Extracting classifier layer weights...")
    final_layer = model.classifier[3]
    weights = final_layer.weight.detach().numpy()
    # weights shape should be (38, 1024) or (38, 576), depending on the exact MobileNet structure

    np.save("cam_weights.npy", weights)
    print(f"Weights saved as cam_weights.npy, shape: {weights.shape}")
    print("Done! Please send 'model_xai.onnx' and 'cam_weights.npy' to the Raspberry Pi.")

if __name__ == "__main__":
    main()
