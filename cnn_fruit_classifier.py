"""
Project 2: Fruit Image Classifier using Convolutional Neural Networks (CNN)
Framework: PyTorch & torchvision

Features:
  - Custom deep CNN architecture with Conv2D, BatchNorm, Dropout, AdaptiveAvgPool
  - Data augmentation (flips, rotations, color jitter) via torchvision transforms
  - Training & Validation loop with learning rate scheduling
  - Performance curves (Loss & Accuracy), Confusion Matrix
  - Visual prediction grid on real test images with confidence scores
  - Model serialization (.pth) for standalone inference
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class FruitCNN(nn.Module):
    """
    3-Stage Convolutional Neural Network for Fruit Classification.
    Input size: (B, 3, 64, 64) -> Output: (B, num_classes)
    """
    def __init__(self, num_classes=3):
        super(FruitCNN, self).__init__()

        # Conv Block 1: 64x64 -> 32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 2: 32x32 -> 16x16
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 3: 16x16 -> 8x8
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully Connected Head
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def get_data_loaders(dataset_dir: str, batch_size=32):
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    # If dataset doesn't exist, trigger generator
    if not os.path.exists(train_dir):
        print(f"Dataset not found at {dataset_dir}. Generating synthetic fruit images...")
        from dataset_generator import build_fruit_dataset
        build_fruit_dataset(dataset_dir, n_train_per_class=120, n_test_per_class=35)

    # Data Augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Deterministic transform for testing / evaluation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    return train_loader, test_loader, class_names, test_dataset


def train_model(model, train_loader, test_loader, device, epochs=12, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    print("\n--- Starting CNN Model Training ---")
    print(f"Device: {device} | Total Epochs: {epochs} | Batch Size: {train_loader.batch_size}")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        scheduler.step()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation / Test Phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = correct_val / total_val

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

    return history


def evaluate_and_plot(model, test_loader, class_names, test_dataset, history, device, output_dir):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    print("\n=======================================================")
    print("      EVALUATION: CNN FRUIT CLASSIFIER REPORT          ")
    print("=======================================================")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    # Create 4-panel evaluation figure
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("Project 2: Fruit Image Classification - CNN Performance Dashboard", fontsize=15, fontweight="bold", y=0.98)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Panel 1: Loss curves
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", lw=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#ff7f0e", label="Val / Test Loss", lw=2)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Cross Entropy Loss", fontweight="bold")
    ax1.set_title("Training & Validation Loss Curve", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Accuracy curves
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="#2ca02c", label="Train Accuracy", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", color="#d62728", label="Val / Test Accuracy", lw=2)
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontweight="bold")
    ax2.set_title("Training & Validation Accuracy Curve", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Confusion Matrix
    ax3 = plt.subplot(2, 2, 3)
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax3, cmap="Oranges", colorbar=False)
    ax3.set_title(f"Confusion Matrix (Test Set, N={len(all_targets)})", fontweight="bold")

    # Panel 4: Visual Test Sample Predictions (Grid of 6 sample images)
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")
    ax4.set_title("Sample Test Predictions (Pred vs Actual)", fontweight="bold")

    # Pick 6 evenly spaced sample images from test dataset
    sample_indices = np.linspace(0, len(test_dataset) - 1, 6, dtype=int)
    inner_grid = fig.add_subfigure(fig.add_gridspec(2, 2)[1, 1])
    sample_axes = inner_grid.subplots(2, 3)
    inner_grid.suptitle("Sample Test Predictions (Pred vs Actual)", fontweight="bold", fontsize=11)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx, ax_sub in zip(sample_indices, sample_axes.flatten()):
        img_tensor, true_label = test_dataset[idx]
        with torch.no_grad():
            out = model(img_tensor.unsqueeze(0).to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx] * 100

        # Unnormalize image for visualization
        np_img = img_tensor.permute(1, 2, 0).numpy()
        np_img = std * np_img + mean
        np_img = np.clip(np_img, 0, 1)

        ax_sub.imshow(np_img)
        is_correct = (pred_idx == true_label)
        color = "green" if is_correct else "red"
        ax_sub.set_title(f"P: {class_names[pred_idx]} ({confidence:.0f}%)\nA: {class_names[true_label]}",
                         fontsize=8, color=color, fontweight="bold")
        ax_sub.axis("off")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fruit_classifier_metrics.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Generated CNN metrics and prediction dashboard: {plot_path}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    train_loader, test_loader, class_names, test_dataset = get_data_loaders(dataset_dir, batch_size=32)
    print(f"Detected Fruit Classes: {class_names}")

    model = FruitCNN(num_classes=len(class_names)).to(device)

    # Train CNN
    history = train_model(model, train_loader, test_loader, device=device, epochs=12, lr=1e-3)

    # Save trained checkpoint
    model_path = os.path.join(current_dir, "fruit_cnn_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, model_path)
    print(f"\nSaved trained CNN checkpoint to: {model_path}")

    # Evaluate & produce dashboard plot
    evaluate_and_plot(model, test_loader, class_names, test_dataset, history, device, output_dir=current_dir)


if __name__ == "__main__":
    main()
