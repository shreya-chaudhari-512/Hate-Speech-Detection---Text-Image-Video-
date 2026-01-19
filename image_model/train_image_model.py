import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- CONFIG ----------------
BATCH_SIZE = 8
EPOCHS = 10
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🖥️ Using device: {DEVICE}")

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
class HateImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel_path = self.data.iloc[idx]["image_path"]
        label = int(self.data.iloc[idx]["label"])

        img_path = os.path.join("dataset/images", img_rel_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# ---------------- LOAD DATA ----------------
train_ds = HateImageDataset("dataset/train.csv", train_tfms)
val_ds = HateImageDataset("dataset/val.csv", val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# ---------------- TRAINING SETUP ----------------
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(DEVICE))
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# ---------------- TRAIN LOOP ----------------
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

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f}")

print("✅ Training complete")

# ---------------- VALIDATION ----------------
model.eval()
all_preds = []
all_labels = []

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

# ---------------- SAVE MODEL ----------------
os.makedirs("image_model", exist_ok=True)
torch.save(model.state_dict(), "image_model/image_model.pth")

print("💾 Model saved at image_model/image_model.pth")
