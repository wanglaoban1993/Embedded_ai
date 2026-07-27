# import os
# import time
# from copy import deepcopy

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import torchvision.models as models

# # 1. Load the pretrained model
# # weights='DEFAULT' is equivalent to pretrained=True, loads weights trained on a large-scale dataset
# model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# # 2. Freeze most parameters
# # Tell the computer: don't touch these weights during training, they're already smart
# for param in model.parameters():
#     param.requires_grad = False

# # 3. Modify the classifier head
# # Looking at the original model structure, the last part is called classifier
# # We need to override it
# num_classes = 7  # Number of disease classes
# input_features = model.classifier[3].in_features # Number of features going into the last layer

# model.classifier[3] = nn.Linear(input_features, num_classes)

# # 4. Move the model to GPU (if available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# print("Model modification complete! It now outputs 7 classes instead of 1000.")

# # --- Standard training loop follows ---
# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# # Note: the optimizer only optimizes classifier parameters, since everything else is frozen
# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# # ----------------- Dataset and DataLoader -----------------
# # Expected data directory structure:
# # data_root/
# #   train/class1/xxx.jpg
# #   train/class2/yyy.jpg
# #   val/class1/zzz.jpg
# #   val/class2/aaa.jpg
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models


def main() -> None:
    # 1. Load the pretrained model
    # weights='DEFAULT' is equivalent to pretrained=True, loads weights trained on a large-scale dataset
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # 2. Freeze most parameters
    # Tell the computer: don't touch these weights during training, they're already smart
    for param in model.parameters():
        param.requires_grad = False

    # 3. Modify the classifier head
    # Looking at the original model structure, the last part is called classifier
    # We need to override it
    num_classes = 7  # Number of disease classes
    input_features = model.classifier[3].in_features  # Number of features going into the last layer

    model.classifier[3] = nn.Linear(input_features, num_classes)

    # 4. Move the model to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print('Device:', device)

    print("Model modification complete! It now outputs 7 classes instead of 1000.")

    # --- Standard training loop follows ---
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Note: the optimizer only optimizes classifier parameters, since everything else is frozen
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    # ----------------- Dataset and DataLoader -----------------
    # Expected data directory structure:
    # data_root/
    #   train/class1/xxx.jpg
    #   train/class2/yyy.jpg
    #   val/class1/zzz.jpg
    #   val/class2/aaa.jpg
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    print(f"Base directory: {base_dir}")
    data_root = os.path.join(project_root, "data", "data_split")  # TODO: change to your data path
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    # Multiprocessing causes errors on Windows, use num_workers=0 for now
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # ----------------- training loop -----------------
    num_epochs = 30
    best_acc = 0.0
    best_state = deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        start_time = time.time()

        # ---- training ----
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels).item()

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects / len(val_dataset)

        # ---- record best model ----
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_state = deepcopy(model.state_dict())

        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    # Save the best model
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(project_root, "models", "mobilenet_v3_small_7cls.pth"))
    print(f"training finished, verify best accurancy: {best_acc:.4f}")


if __name__ == "__main__":
    main()