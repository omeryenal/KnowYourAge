import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for part in os.listdir(root_dir):
            part_path = os.path.join(root_dir, part)
            if not os.path.isdir(part_path):
                continue
            for file in os.listdir(part_path):
                if file.endswith(".jpg"):
                    self.image_paths.append(os.path.join(part_path, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)
        age = int(filename.split("_")[0])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)
