import sys, os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Path configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import UTKFaceDataset

# Image visualization function
def imshow(img_tensor, age):
    img = img_tensor.numpy().transpose((1, 2, 0))  # [C,H,W] → [H,W,C]
    img = (img * 0.5) + 0.5  # Undo normalization
    plt.imshow(img)
    plt.title(f"True Age: {int(age * 100)}")  # 🔁 Rescale age back
    plt.axis("off")
    plt.show()

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset and DataLoader
# Set the absolute path based on your current file location
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # KnowYourAge/
DATA_DIR = os.path.join(BASE_DIR, "data", "UTKFace")

dataset = UTKFaceDataset(root_dir=DATA_DIR, transform=transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Display one sample image
for images, ages in dataloader:
    imshow(images[0], ages[0].item())
    break
