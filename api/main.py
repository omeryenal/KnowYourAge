from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.baseline import AgeRegressionCNN

# 🚀 App başlat
app = FastAPI()

# 🧠 Model yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeRegressionCNN().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()

# 🔄 Görsel için transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.post("/predict")
async def predict_age(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            predicted_age = output.item()

        return JSONResponse(content={"predicted_age": round(predicted_age, 2)})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
