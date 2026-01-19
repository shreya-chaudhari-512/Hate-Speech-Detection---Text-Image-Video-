import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet18_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("image_model/image_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_symbol(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    return "HATE" if pred == 1 else "NON-HATE"
