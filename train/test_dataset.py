# train/test_dataset.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import UTKFaceDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Tensor'dan imaj gösterimi
def imshow(img_tensor, age):
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = (img * 0.5) + 0.5  # normalize edilmiş veriyi geri al
    plt.imshow(img)
    plt.title(f"Predicted Age: {age}")
    plt.axis("off")
    plt.show()

# Transformlar
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset ve DataLoader
dataset = UTKFaceDataset(root_dir="../data/UTKFace", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Örnek göster
for images, ages in dataloader:
    imshow(images[0], ages[0].item())
    break
