import argparse
import os
import time
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, num_classes)
    return model.to(device)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def get_loaders(data_root: str, batch_size: int, num_workers: int, image_size: int):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_tf, eval_tf = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, len(train_dataset), len(val_dataset), train_dataset.classes


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    ce = F.cross_entropy(student_logits, targets)
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)
    return alpha * ce + (1.0 - alpha) * kd


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a MobileNetV3-Small student from a teacher")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    parser.add_argument(
        "--data-root",
        default=os.path.join(project_root, "data", "data_split"),
        help="Path to dataset root with train/ and val/ subfolders",
    )
    parser.add_argument(
        "--teacher-weights",
        default=os.path.join(project_root, "models", "mobilenet_v3_small_7cls.pth"),
        help="Path to teacher weights (.pth)",
    )
    parser.add_argument(
        "--student-weights",
        default=os.path.join(project_root, "models", "mobilenet_v3_small_7cls_distilled.pth"),
        help="Output path for distilled student weights",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for CE loss vs KD loss")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--num-classes", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root not found: {args.data_root}")
    if not os.path.isfile(args.teacher_weights):
        raise FileNotFoundError(f"teacher weights not found: {args.teacher_weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_size, val_size, classes = get_loaders(
        args.data_root, args.batch_size, args.num_workers, args.image_size
    )
    num_classes = args.num_classes or len(classes)

    teacher = build_model(num_classes=num_classes, device=device)
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
    teacher.eval()

    student = build_model(num_classes=num_classes, device=device)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    best_acc = 0.0
    best_state = deepcopy(student.state_dict())

    for epoch in range(args.epochs):
        start_time = time.time()
        student.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)
            loss = distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=labels,
                alpha=args.alpha,
                temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(student_logits, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / train_size

        val_acc = evaluate(student, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = deepcopy(student.state_dict())

        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s"
        )

    student.load_state_dict(best_state)
    torch.save(student.state_dict(), args.student_weights)
    print(f"Distillation finished. Best Val Acc: {best_acc:.4f}")
    print(f"Saved student weights to: {args.student_weights}")


if __name__ == "__main__":
    main()
