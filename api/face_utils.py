import numpy as np
import cv2
from PIL import Image

# Load OpenCV's Haar cascade for face detection
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_and_crop_face(pil_img):
    """
    Detects and crops the largest face in a given PIL image.
    If no face is found, the original image is returned.

    Args:
        pil_img (PIL.Image): Input image in RGB

    Returns:
        PIL.Image: Cropped face image or original if no face found
    """
    # Convert PIL → NumPy → OpenCV BGR
    np_rgb = np.array(pil_img.convert("RGB"))
    np_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

    # Convert to grayscale and detect faces
    gray = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(64, 64),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return pil_img  # No face found

    # Select the largest detected face (most likely main subject)
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    # Apply 10% margin around face bounding box
    margin_x = int(0.1 * w)
    margin_y = int(0.1 * h)
    x1 = max(x - margin_x, 0)
    y1 = max(y - margin_y, 0)
    x2 = min(x + w + margin_x, np_bgr.shape[1])
    y2 = min(y + h + margin_y, np_bgr.shape[0])

    # Crop and convert back to RGB PIL image
    cropped_bgr = np_bgr[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)
