import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from model import CNN  


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Model
# -----------------------------
model = CNN().to(device)

model.load_state_dict(
    torch.load("best_mnist_cnn.pth", map_location=device, weights_only=True)
)

model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# -----------------------------
# Test Dataset
# -----------------------------
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)


# -----------------------------
# Inference + Metrics
# -----------------------------
all_preds = []
all_labels = []
all_confidences = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())


# -----------------------------
# Accuracy
# -----------------------------
acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {acc*100:.2f}%")



# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))


# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# -----------------------------
# Visualize Predictions
# -----------------------------
images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    confidences, preds = torch.max(probs, dim=1)

plt.figure(figsize=(12,6))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    plt.title(
        f"Pred: {preds[i].item()} ({confidences[i]*100:.1f}%)\nTrue: {labels[i].item()}"
    )
    plt.axis('off')

plt.tight_layout()
plt.show()
