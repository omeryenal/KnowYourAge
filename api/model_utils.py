import os
import torch
from torchvision import transforms
from train.train_dex import DEXAgeModel, NUM_BINS

def load_model():
    """
    Loads the pretrained DEXAgeModel and selects the appropriate device (CUDA, MPS, CPU).

    Returns:
        model (torch.nn.Module): Loaded PyTorch model
        device (torch.device): Device used for inference
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )

    model = DEXAgeModel(num_bins=NUM_BINS, pretrained_dex=False).to(device)
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "dex_utkface.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device


# Preprocessing function for DEX input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(pil_image):
    """
    Applies preprocessing to the image before feeding it to the model.

    Args:
        pil_image (PIL.Image): Input face image

    Returns:
        torch.Tensor: Preprocessed tensor [1, 3, 224, 224]
    """
    return transform(pil_image).unsqueeze(0)
