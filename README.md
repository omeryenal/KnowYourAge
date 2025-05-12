# 🧠 KnowYourAge

**KnowYourAge** is a fully custom, from-scratch AI system that predicts how old someone *looks* — not how old they *are* — using live webcam input and deep learning.

Unlike traditional projects that rely on pre-trained models or APIs, KnowYourAge is built entirely from the ground up. The model architecture, training process, API, and frontend are all handcrafted for maximum transparency and learning value.

---

## 🔍 What It Does

- Opens your webcam in real time
- Captures your face image
- Feeds it into a neural network
- Predicts your *apparent* age (how old you look)
- Displays the result instantly on the web interface

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|------------|
| Model Training | PyTorch (custom CNN) |
| Dataset | [UTKFace](https://susanqq.github.io/UTKFace/) |
| Backend API | FastAPI (Python) |
| Frontend | HTML, JavaScript (Vanilla) |
| Camera Access | WebRTC (`getUserMedia`) |
| Image Preprocessing | Pillow, OpenCV |
| Model Deployment | Local (with option to Dockerize later) |

---

## 📁 Project Structure

know-your-age/
├── data/ # Raw and preprocessed face images
├── model/ # CNN model definition and saved weights
├── train/ # Model training script
├── api/ # FastAPI backend server
├── frontend/ # HTML + JS camera interface
├── README.md # You are here
└── venv/ # Python virtual environment

---

## 🚧 Status

🔨 **In Development** — This project is part of a 50-day full-stack AI build journey.

---

## ✨ Goals

- [ ] Train a fully custom CNN to predict age from face images  
- [ ] Serve the model via FastAPI  
- [ ] Build a web interface that connects webcam → prediction  
- [ ] Ensure everything runs locally with no third-party ML APIs  
- [ ] Maintain transparency, reproducibility, and educational value  

---

## 📸 Example Output (Coming Soon)

> Age Prediction: **24.3 years**  
> _"You look 24!"_

---

## 🤝 Author

**Ömer Yenal**  
Industrial Engineering @ Koç University  
AI & ML Enthusiast | Full-stack learner  
[GitHub](https://github.com/omeryenal)

---

