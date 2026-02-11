import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from model import CNN


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# -----------------------------
# Data Loading (Train + Validation ONLY)
# -----------------------------
def load_data(batch_size=256):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # ⭐ Split into Train / Validation
    train_size = int(0.85 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=generator
)


    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


# -----------------------------
# Validation Function
# -----------------------------
def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0
    running_loss = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    loss = running_loss / len(loader)

    return loss, accuracy


# -----------------------------
# Training Function
# -----------------------------
def train_model(epochs=10, lr=0.001):

    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = load_data()

    model = CNN().to(device)

    print("Total parameters:",
          sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_val_acc = 0

    # Early stopping setup
    patience = 3
    counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # TRAIN ACCURACY
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"""
Epoch [{epoch+1}/{epochs}]
Train Loss: {train_loss:.4f}
Train Accuracy: {train_acc:.2f}%
Val Loss: {val_loss:.4f}
Val Accuracy: {val_acc:.2f}%
LR: {optimizer.param_groups[0]['lr']:.6f}
""")

        # Save best model
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "best_mnist_cnn.pth")
            best_val_acc = val_acc
            counter = 0
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print("Early stopping triggered.")
            break

    print("Best Validation Accuracy:", best_val_acc)

    # -----------------------------
    # Visualization
    # -----------------------------
    plt.figure(figsize=(16,5))

    # Loss curves
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy curves
    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(val_accuracies, label="Val Accuracy", marker='o')
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    train_model()
