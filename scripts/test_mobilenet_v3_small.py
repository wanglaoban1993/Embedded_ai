import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models


def main() -> None:
    # 1. Load the pretrained model structure
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # 2. Modify the classifier head
    num_classes = 7
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, num_classes)

    # 3. Load the trained weights
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    weights_path = os.path.join(project_root, "models", "mobilenet_v3_small_7cls.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("Device:", device)

    # 4. Build the test set DataLoader
    data_root = os.path.join(project_root, "data", "data_split")
    test_dir = os.path.join(data_root, "test")

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 5. Test set accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()