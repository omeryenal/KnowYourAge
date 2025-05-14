from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import sys

# Model import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.baseline import AgeRegressionCNN

# ⚙️ Device (M2 Max uyumlu)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 🎯 FastAPI app
app = FastAPI()

# 📦 Model yükle
model = AgeRegressionCNN().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()

# 🔁 Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 📥 Base64 veri modeli
class ImagePayload(BaseModel):
    image_base64: str

@app.post("/predict")
def predict_base64(payload: ImagePayload):
    try:
        # Base64 çözümle
        image_data = base64.b64decode(payload.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            predicted_age = output.item()

        return {"predicted_age": round(predicted_age, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
