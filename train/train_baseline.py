import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader, random_split
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

# Load full dataset
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "UTKFace")
dataset = UTKFaceDataset(root_dir=DATA_DIR, transform=transform)

# Split into train/val
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Model
model = AgeRegressionCNN().to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
mae_loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, ages in train_loader:
        images, ages = images.to(device), ages.to(device)
        outputs = model(images)
        loss = criterion(outputs, ages)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for images, ages in val_loader:
            images, ages = images.to(device), ages.to(device)
            outputs = model(images)
            loss = criterion(outputs, ages)
            mae = mae_loss(outputs, ages)
            val_loss += loss.item()
            val_mae += mae.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.2f} | "
          f"Val Loss: {val_loss/len(val_loader):.2f} | Val MAE: {val_mae/len(val_loader):.2f}")

# Örnek tahmin
model.eval()
with torch.no_grad():
    for images, ages in val_loader:
        images = images.to(device)
        output = model(images[0].unsqueeze(0))
        print(f"True age: {ages[0].item()} | Predicted: {output.item():.2f}")
        break
