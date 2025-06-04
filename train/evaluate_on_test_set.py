import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import UTKFaceDataset
from model.baseline import AgeRegressionCNN
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms


# 🔧 Ayarlar
MODEL_PATH = "checkpoints/deep_model.pt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "UTKFace")

# 🧼 Transformlar
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 📦 Dataset
dataset = UTKFaceDataset(root_dir=DATA_DIR, transform=transform)

# 📊 Train / Val / Test split
total_size = len(dataset)
test_size = int(0.10 * total_size)
val_size = int(0.20 * (total_size - test_size))
train_size = total_size - val_size - test_size

_, _, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                  generator=torch.Generator().manual_seed(42))

test_loader = DataLoader(test_dataset, batch_size=1)

# 🧠 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeRegressionCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 🔬 Final MAE hesabı
mae_loss = torch.nn.L1Loss()
total_mae = 0.0

with torch.no_grad():
    for images, ages in test_loader:
        images, ages = images.to(device), ages.to(device)
        outputs = model(images)
        total_mae += mae_loss(outputs, ages).item()

avg_mae = total_mae / len(test_loader)
print(f"🧪 Final Hold-Out Test MAE: {avg_mae:.2f}")
