import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import UTKFaceDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def imshow(img_tensor, age):
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = (img * 0.5) + 0.5
    plt.imshow(img)
    plt.title(f"True Age: {age}")
    plt.axis("off")
    plt.show()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

dataset = UTKFaceDataset(root_dir="../data/UTKFace", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for images, ages in dataloader:
    imshow(images[0], ages[0].item())
    break
