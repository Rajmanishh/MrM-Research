import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= MODEL =================
model = CNN().to(device)
model.load_state_dict(torch.load("best_mnist_cnn.pth", map_location=device))
model.eval()

# ================= IMPORTANT =================
# MUST match training preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ================= TEST DATA =================
test_loader = DataLoader(
    datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    ),
    batch_size=1,
    shuffle=True
)

# ================= TEST LOOP =================
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)

        total += 1
        if predicted.item() == labels.item():
            correct += 1

        # test on first 100 samples
        if total == 100:
            break

print(f"\nAccuracy: {correct}/100")