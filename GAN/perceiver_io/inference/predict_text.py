import torch
import torch.nn as nn

from models.perceiver_io import PerceiverIO
from models.heads.classification_head import ClassificationHead


# 🔹 Config (MATCH TRAINING EXACTLY)
EMBED_DIM = 128
NUM_CLASSES = 14
MAX_LEN = 150   # ✅ MUST match training

# 🔥 Force GPU
if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU not available")

DEVICE = torch.device("cuda")

print("🚀 Using GPU:", torch.cuda.get_device_name(0))


# 🔹 Model
embedding = nn.Embedding(256, EMBED_DIM).to(DEVICE)

model = PerceiverIO(C=EMBED_DIM, N=64, D=128, num_layers=4).to(DEVICE)
head = ClassificationHead(D=128, num_classes=NUM_CLASSES).to(DEVICE)


# 🔹 Load weights
checkpoint = torch.load("best_model.pth", map_location=DEVICE)

model.load_state_dict(checkpoint["model"])
head.load_state_dict(checkpoint["head"])
embedding.load_state_dict(checkpoint["embedding"])

model.eval()
head.eval()
embedding.eval()


# 🔹 DBpedia labels
labels = [
    "Company", "Educational Institution", "Artist", "Athlete",
    "Office Holder", "Mean of Transportation", "Building",
    "Natural Place", "Village", "Animal",
    "Plant", "Album", "Film", "Written Work"
]


# 🔥 UTF-8 ENCODING
def encode(text):
    text = text.lower().strip()

    byte_ids = list(text.encode("utf-8"))

    if len(byte_ids) < MAX_LEN:
        byte_ids += [0] * (MAX_LEN - len(byte_ids))
    else:
        byte_ids = byte_ids[:MAX_LEN]

    return torch.tensor(byte_ids, dtype=torch.long).unsqueeze(0)


# 🔥 Prediction function
def predict(text):
    x = encode(text).to(DEVICE)
    x = embedding(x)

    with torch.no_grad():
        # 🔥 Faster inference with autocast
        with torch.autocast(device_type="cuda"):
            latents = model(x)
            logits = head(latents)

        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = torch.topk(probs, k=3)

    return top_idx[0], top_prob[0]


# 🔥 LIVE TEST LOOP
while True:
    text = input("\nEnter text (or 'quit'): ")

    if text.lower() == "quit":
        break

    top_idx, top_prob = predict(text)

    print("\n🔍 Predictions:")
    for i in range(len(top_idx)):
        label = labels[top_idx[i]]
        confidence = top_prob[i].item()
        print(f"{i+1}. {label} ({confidence:.2f})")

    # 🔥 Confidence warning
    if top_prob[0].item() < 0.4:
        print("⚠️ Low confidence prediction")