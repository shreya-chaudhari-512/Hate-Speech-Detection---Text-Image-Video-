import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- CONFIG ----------------
BATCH_SIZE = 8
EPOCHS = 10
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🖥️ Using device: {DEVICE}")

DATA_DIR = "dataset/images/image_symbol"

# ---------------- TRANSFORMS ----------------
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------------- DATASET ----------------
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfms)

# Train / Val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("📂 Classes:", full_dataset.classes)

# ---------------- MODEL ----------------
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# ---------------- TRAINING ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f}")

print("✅ Training complete")

# ---------------- VALIDATION ----------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n📊 Validation Accuracy:", round(acc, 3))
print("📉 Confusion Matrix:")
print(cm)

# ---------------- SAVE ----------------
os.makedirs("image_model", exist_ok=True)
torch.save(model.state_dict(), "image_model/image_model.pth")

print("💾 Model saved at image_model/image_model.pth")
