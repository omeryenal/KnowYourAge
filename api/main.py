import os
import sys
import io
import base64

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import torch

# 🔁 Relative imports for utils inside api/
from .model_utils import load_model, preprocess_image
from .face_utils import detect_and_crop_face

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ADMIN_URL = os.getenv("ADMIN_URL", "http://localhost:3001")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, ADMIN_URL],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Load model and device once at startup
model, device = load_model()

# Request body schema
class ImagePayload(BaseModel):
    image_base64: str  # base64-encoded image string

@app.post("/predict")
def predict_base64(payload: ImagePayload):
    """
    Accepts a base64-encoded image, detects the face,
    and returns the predicted age as JSON.

    Example Request:
        {
            "image_base64": "<base64 string>"
        }

    Response:
        {
            "predicted_age": 25
        }
    """
    try:
        # Decode base64 and convert to PIL
        image_data = base64.b64decode(payload.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Crop face
        face_image = detect_and_crop_face(image)

        # Preprocess and predict
        image_tensor = preprocess_image(face_image).to(device)
        with torch.no_grad():
            age_pred = model(image_tensor)
            predicted_age = int(age_pred.item())

        return {"predicted_age": predicted_age}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run via: uvicorn api.main:app --reload
