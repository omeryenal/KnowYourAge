import cv2
import base64
import requests
import time

API_URL = "http://localhost:8000/predict"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kameraya erişilemiyor.")
    exit()

print("Press 'q' to quit.")
last_prediction_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame alınamadı.")
        break

    current_time = time.time()
    if current_time - last_prediction_time >= 5:  # her 5 saniyede bir tahmin
        resized_frame = cv2.resize(frame, (64, 64))
        _, buffer = cv2.imencode('.jpg', resized_frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        data = {"image_base64": img_base64}

        try:
            response = requests.post(API_URL, json=data)
            if response.status_code == 200:
                print("🧠 Tahmin edilen yaş:", response.json().get("predicted_age"))
            else:
                print("❌ API hatası:", response.status_code, response.text)
        except Exception as e:
            print("❌ API bağlantı hatası:", e)

        last_prediction_time = current_time

    cv2.imshow("Webcam - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
