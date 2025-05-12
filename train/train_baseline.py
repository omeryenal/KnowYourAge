# train/train_baseline.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from dataset import UTKFaceDataset
from model.baseline import AgeRegressionCNN
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go from /train to root
DATA_DIR = os.path.join(BASE_DIR, "data", "UTKFace")

dataset = UTKFaceDataset(root_dir=DATA_DIR, transform=transform)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers =0)

# Model
model = AgeRegressionCNN().to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, ages in train_loader:
        images = images.to(device)
        

        ages = ages.float().to(device)

        outputs = model(images)
       
        loss = criterion(outputs, ages)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")
model.eval()
with torch.no_grad():
    for images, ages in train_loader:
        images = images.to(device)
        ages = ages.to(device)

        output = model(images[0].unsqueeze(0))  # sadece 1 görsel

        print(f"True age: {ages[0].item()} | Predicted age: {output.item():.2f}")
        break