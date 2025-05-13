import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.makedirs("checkpoints", exist_ok=True)
MODEL_PATH = "checkpoints/best_model.pt"

import torch
from torch.utils.data import DataLoader, random_split
from dataset import UTKFaceDataset
from model.baseline import AgeRegressionCNN
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformlar
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "UTKFace")
dataset = UTKFaceDataset(root_dir=DATA_DIR, transform=transform)

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Model, loss, optimizer
model = AgeRegressionCNN().to(device)
criterion = nn.MSELoss()
mae_loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Scheduler & early stopping
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, 
)
early_stop_counter = 0
early_stop_patience = 5

# Eğitim döngüsü
EPOCHS = 50
best_mae = float("inf")

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

    # Validasyon
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

    val_mae_epoch = val_mae / len(val_loader)
    scheduler.step(val_mae_epoch)

    if val_mae_epoch < best_mae:
        best_mae = val_mae_epoch
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Best model saved at epoch {epoch+1} with Val MAE: {best_mae:.2f}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"⚠️ No improvement. Early stop patience: {early_stop_counter}/{early_stop_patience}")
        if early_stop_counter >= early_stop_patience:
            print("🛑 Early stopping triggered.")
            break

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.2f} | "
          f"Val Loss: {val_loss/len(val_loader):.2f} | Val MAE: {val_mae_epoch:.2f}")

# Örnek tahmin
model.eval()
with torch.no_grad():
    for images, ages in val_loader:
        images = images.to(device)
        output = model(images[0].unsqueeze(0))
        print(f"True age: {ages[0].item()} | Predicted: {output.item():.2f}")
        break
