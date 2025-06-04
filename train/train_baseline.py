import sys
import os

# ── Add the project root to Python path ──
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, os.pardir, os.pardir))
sys.path.append(project_root)

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import UTKFaceDataset
from model.deep_cnn import DeepAgeCNN  # Updated model

# ─────────── Device Selection ───────────
use_cuda = torch.cuda.is_available()
device = torch.device(
    "cuda" if use_cuda
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

# ─────────── Hyperparameters ───────────
BATCH_SIZE_SMALL = 32       # Smaller batch size for faster warm-up
BATCH_SIZE_LARGE = 64       # Larger batch for more stable training
LR_BASE = 1e-4              # Base learning rate
WEIGHT_DECAY = 5e-5
EPOCHS = 100
EARLY_STOP_PATIENCE = 8     # Early stopping patience

# ─────────── Path Configuration ───────────
os.makedirs(os.path.join(project_root, "checkpoints"), exist_ok=True)
MODEL_PATH = os.path.join(project_root, "checkpoints", "deep_model.pt")
DATA_DIR = os.path.join(project_root, "data", "UTKFace")

# ─────────── Data Transforms ───────────

# Light training augmentations (ColorJitter & RandomErasing with p=0.1)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Only resize and normalize for validation
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

def main():
    # ───── Load Dataset & Split into Train/Validation ─────
    dataset = UTKFaceDataset(root_dir=DATA_DIR, transform=None)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # ───── Model, Loss, Optimizer ─────
    model = DeepAgeCNN().to(device)
    criterion = nn.SmoothL1Loss()  # Similar to Huber loss
    mae_loss = nn.L1Loss()         # For MAE tracking

    optimizer = optim.Adam(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_mae = float("inf")
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        # ───── Adjust Learning Rate & Batch Size ─────
        current_lr = optimizer.param_groups[0]["lr"]
        batch_size = BATCH_SIZE_SMALL if epoch < 5 else BATCH_SIZE_LARGE

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_cuda
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_cuda
        )

        # ───── Training Phase ─────
        model.train()
        running_train_loss = 0.0
        for images, ages in train_loader:
            images, ages = images.to(device), ages.to(device)

            outputs = model(images)
            loss = criterion(outputs, ages)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)

        # ───── Validation Phase ─────
        model.eval()
        running_val_loss = 0.0
        running_val_mae = 0.0
        with torch.no_grad():
            for images, ages in val_loader:
                images, ages = images.to(device), ages.to(device)
                outputs = model(images)
                v_loss = criterion(outputs, ages)
                v_mae = mae_loss(outputs, ages)

                running_val_loss += v_loss.item() * images.size(0)
                running_val_mae += v_mae.item() * images.size(0)

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        avg_val_mae = running_val_mae / len(val_loader.dataset)
        scheduler.step(avg_val_mae)

        # ───── Best MAE Checkpoint + Early Stopping ─────
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            torch.save(model.state_dict(), MODEL_PATH)
            early_stop_counter = 0
            print(f"✅ Epoch {epoch+1}: New best Val MAE = {(best_mae * 100):.2f} years")
        else:
            early_stop_counter += 1
            print(
                f"⚠️ Epoch {epoch+1}: No improvement "
                f"(patience {early_stop_counter}/{EARLY_STOP_PATIENCE}). "
                f"Val MAE = {(avg_val_mae * 100):.2f} years"
            )
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print("🛑 Early stopping triggered.")
                break

        # ───── Epoch Summary Output ─────
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss = {(avg_train_loss * 100):.2f} | "
            f"Val Loss   = {(avg_val_loss * 100):.2f} | "
            f"Val MAE    = {(avg_val_mae * 100):.2f} | "
            f"LR = {optimizer.param_groups[0]['lr']:g} | "
            f"BS = {batch_size}"
        )

    print(f"\nTraining complete. Best Val MAE = {(best_mae * 100):.2f} years")

    # ───── Sample Prediction ─────
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        with torch.no_grad():
            sample_images, sample_ages = next(iter(val_loader))
            sample_images = sample_images.to(device)
            pred = model(sample_images[0].unsqueeze(0))
            print(
                f"🔬 Actual Age: {(sample_ages[0].item() * 100):.1f} | "
                f"Predicted: {(pred.item() * 100):.1f}"
            )

if __name__ == "__main__":
    main()
