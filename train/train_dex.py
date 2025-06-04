import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random

# ── Add project root to Python path ──
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, os.pardir, os.pardir))
sys.path.append(project_root)

# 1) Import CleanUTKFaceDataset
from train.dataset import CleanUTKFaceDataset as UTKFaceDataset

# Update this path if you downloaded the official DEX weights
DEX_PRETRAINED_PATH = os.path.join(project_root, "checkpoints", "dex_pretrained.pth")

# ------------- Device Selection -------------
use_cuda = torch.cuda.is_available()
device = torch.device(
    "cuda" if use_cuda
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

# ------------- Constants / Hyperparameters -------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BATCH_SIZE = 32
EPOCHS = 20           # Usually 3–5 epochs of fine-tuning is enough
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 5          # Patience value for ReduceLROnPlateau

# Age range is 0–100 → num_bins = 101
NUM_BINS = 101

# Dataset and checkpoint paths
DATA_DIR = os.path.join(project_root, "data", "UTKFace")
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "dex_utkface.pt")


# ------------------------ DEX Age Model (VGG-16 Based) ------------------------
class DEXAgeModel(nn.Module):
    """
    DEX VGG-16-based model:
    - VGG-16 backbone (pretrained on ImageNet or DEX)
    - Final layer outputs age distribution (101 classes: 0–100)
    - Age prediction via expectation over softmax
    """

    def __init__(self, num_bins=NUM_BINS, pretrained_dex=False):
        super(DEXAgeModel, self).__init__()

        # 1) Load pretrained VGG-16
        self.backbone = torch.hub.load(
            'pytorch/vision:v0.15.2',
            'vgg16',
            pretrained=True
        )

        # 2) Modify the classifier to output num_bins
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_bins)
        )

        # Optionally load pretrained DEX weights
        if pretrained_dex:
            if not os.path.exists(DEX_PRETRAINED_PATH):
                raise FileNotFoundError(f"DEX pretrained weights not found: {DEX_PRETRAINED_PATH}")
            state = torch.load(DEX_PRETRAINED_PATH, map_location=device)
            self.backbone.load_state_dict(state)
            print(f"[DEX] Pretrained DEX weights loaded → {DEX_PRETRAINED_PATH}")

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        → backbone(x)  → [B, num_bins] logits
        → softmax → [B, num_bins] probabilities
        → expectation = sum_i p_i * i → [B] (age prediction [0–100])
        """
        logits = self.backbone(x)
        probs = torch.softmax(logits, dim=1)
        idxs = torch.arange(0, logits.size(1), device=logits.device, dtype=torch.float32)
        ages = torch.sum(probs * idxs.unsqueeze(0), dim=1)
        return ages


# ------------------------ Dataset & Dataloader ------------------------
def create_dataloaders(batch_size=BATCH_SIZE):
    """
    Loads CleanUTKFaceDataset, splits into 80% train / 20% val,
    and returns DataLoaders for both.
    """
    dataset = UTKFaceDataset(root_dir=DATA_DIR, transform=None)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda
    )

    return train_loader, val_loader


# ------------------------ Training Function ------------------------
def train_dex():
    train_loader, val_loader = create_dataloaders()

    model = DEXAgeModel(num_bins=NUM_BINS, pretrained_dex=False).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=PATIENCE
    )

    best_mae = float("inf")
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        for images, ages_norm in train_loader:
            images = images.to(device)
            ages = ages_norm.to(device) * 100.0

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, ages)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_mae = 0.0
        with torch.no_grad():
            for images, ages_norm in val_loader:
                images = images.to(device)
                ages = ages_norm.to(device) * 100.0

                preds = model(images)
                loss = criterion(preds, ages)
                running_val_loss += loss.item() * images.size(0)
                running_val_mae += torch.abs(preds - ages).sum().item()

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        avg_val_mae = running_val_mae / len(val_loader.dataset)

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Train Loss = {avg_train_loss:.4f} | "
            f"Val Loss = {avg_val_loss:.4f} | "
            f"Val MAE = {avg_val_mae:.2f} years"
        )

        # Save best model
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ Epoch {epoch+1}: Best model saved (MAE = {best_mae:.2f} years)")

    print(f"\nTraining complete. Best Val MAE = {best_mae:.2f} years")
    print(f"Checkpoint saved at → {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train_dex()
