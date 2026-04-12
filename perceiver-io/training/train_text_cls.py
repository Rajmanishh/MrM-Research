import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import seaborn as sns

from models.perceiver_io import PerceiverIO
from models.heads.classification_head import ClassificationHead
from data.loaders.text_loader import TextDataset


def train():

    # 🔹 Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 3e-4
    NUM_CLASSES = 14
    EMBED_DIM = 128
    MAX_LEN = 150   # 🔥 optimized (faster than 200)
    WARMUP_EPOCHS = 2

    # 🔥 FORCE GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU not available")

    DEVICE = torch.device("cuda")

    torch.backends.cudnn.benchmark = True

    # 🔹 Dataset
    train_dataset = TextDataset(split="train", max_len=MAX_LEN)
    val_dataset = TextDataset(split="test", max_len=MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True
    )

    # 🔹 Embedding
    embedding = nn.Embedding(256, EMBED_DIM).to(DEVICE)

    # 🔹 Model
    model = PerceiverIO(C=EMBED_DIM, N=64, D=128, num_layers=4).to(DEVICE)
    head = ClassificationHead(D=128, num_classes=NUM_CLASSES).to(DEVICE)

    # 🔥 Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in head.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in embedding.parameters() if p.requires_grad)
    print(f"\n🚀 Total Trainable Parameters: {total_params/1e6:.2f}M\n")

    # 🔹 Loss + Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(head.parameters()) +
        list(embedding.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    # 🔥 AMP scaler
    scaler = torch.amp.GradScaler(device="cuda")

    # 🔥 Warmup + Cosine
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_acc = 0
    best_epoch = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lr_values = []

    for epoch in range(EPOCHS):

        # ================= TRAIN =================
        model.train()
        head.train()
        embedding.train()

        total_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            x = embedding(x)

            optimizer.zero_grad()

            # 🔥 Mixed precision forward
            with torch.autocast(device_type="cuda"):
                logits = head(model(x))
                loss = criterion(logits, y)

            # 🔥 Scaled backward
            scaler.scale(loss).backward()

            # 🔥 Unscale + clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()),
                1.0
            )

            # 🔥 Step
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # ================= VALIDATION =================
        model.eval()
        head.eval()
        embedding.eval()

        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                x = embedding(x)
                logits = head(model(x))

                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)

                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_values.append(current_lr)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "embedding": embedding.state_dict()
            }, "best_model.pth")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    print(f"\n🔥 Best Validation Accuracy: {best_acc:.4f} (Epoch {best_epoch})")

    # ================= GRAPHS =================

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss"); plt.savefig("loss.png"); plt.show()

    plt.figure()
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend(); plt.title("Accuracy"); plt.savefig("accuracy.png"); plt.show()

    plt.figure()
    plt.plot(lr_values)
    plt.title("Learning Rate Schedule")
    plt.savefig("lr.png")
    plt.show()

    # ================= CONFUSION MATRIX =================
    raw_cm = confusion_matrix(all_labels, all_preds)
    cm = raw_cm.astype('float') / raw_cm.sum(axis=1, keepdims=True)

    labels = [
        "Company", "Educational Institution", "Artist", "Athlete",
        "Office Holder", "Mean of Transportation", "Building",
        "Natural Place", "Village", "Animal",
        "Plant", "Album", "Film", "Written Work"
    ]

    print("\n📊 Per-class Accuracy:")
    per_class_acc = raw_cm.diagonal() / raw_cm.sum(axis=1)
    for i, acc in enumerate(per_class_acc):
        print(f"{labels[i]}: {acc:.2f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.savefig("confusion.png")
    plt.show()

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    train()