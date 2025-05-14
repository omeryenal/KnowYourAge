import base64
import requests



with open("/Users/omeryenal/Desktop/KnowYourAge/data/UTKFace/part2/2_1_1_20170113002906252.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode("utf-8")

data = {
    "image_base64": img_base64
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
