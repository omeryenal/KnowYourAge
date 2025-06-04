import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from dataset import UTKFaceDataset
from model.deep_cnn import DeepAgeCNN
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 🔧 Ayarlar
MODEL_PATH = "checkpoints/deep_model.pt"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "UTKFace")

# 📦 Dataset & Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = UTKFaceDataset(root_dir=DATA_DIR, transform=transform)
_, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# 🔄 Model yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepAgeCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


with torch.no_grad():
    for i, (images, ages) in enumerate(val_loader):
        images = images.to(device)
        outputs = model(images)

        true_age = ages.item() * 100
        predicted_age = outputs.item() * 100

        print(f"🎯 True age: {true_age:.0f} | 🧠 Predicted age: {predicted_age:.0f}")

        if i == 10:
            break
