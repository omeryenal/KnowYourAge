# 🧠 KnowYourAge

**KnowYourAge** is a full-stack AI-powered application that predicts a person’s age based on their face captured from a live webcam stream. It uses a custom-trained deep learning model inspired by the DEX (Deep EXpectation) architecture and includes a modern React-based frontend and a FastAPI backend.

---

## 📌 Project Overview

This project was built completely from scratch with the goal of:

- Building a **deep learning model** for age prediction using real facial data
- Applying **face detection and preprocessing** techniques for better accuracy
- Developing a **real-time prediction interface** using webcam input
- Deploying a clean, open-source, and modular AI product

---

## 🧠 Model Architecture & Training

### 🔨 From Scratch to Production

1. **Dataset**:  
   We used the [UTKFace](https://susanqq.github.io/UTKFace/) dataset, which contains over 20,000 face images labeled by age, gender, and ethnicity.

2. **Cleaning**:  
   - Malformed filenames skipped  
   - Outlier ages removed using 1st and 99th percentiles  
   - Low-resolution images (< 64×64) filtered out  

3. **Model Architecture**:  
   The model is a custom PyTorch CNN inspired by **DEX-VGG16**, trained to regress normalized age (between 0 and 1), then rescaled to [0–100].

4. **Loss & Optimization**:
   - `SmoothL1Loss` for regression
   - `ReduceLROnPlateau` scheduler
   - `Adam` optimizer
   - Early stopping & model checkpointing

---

## 📦 Project Structure

```
KnowYourAge/
├── api/                    # FastAPI backend
│   ├── main.py             # API endpoint for predictions
│   ├── model_utils.py      # Model loading & preprocessing
│   └── face_utils.py       # Face detection logic (OpenCV)
├── checkpoints/            # Saved PyTorch model (.pt)
├── dataset/                # Custom PyTorch Dataset class
├── frontend/               # React frontend (webcam & UI)
│   ├── components/         # UI components (buttons, webcam box)
│   ├── pages/              # App pages (Landing, Predict)
│   └── App.js              # Route management
├── train/                  # Model training pipeline
│   └── train_dex.py        # DEXAgeModel definition & training
└── README.md               # You are here
```

---

## 🔍 Features

- 🎥 **Live Webcam Input**  
  Captures a webcam image with a single click and sends it to the backend.

- 🧠 **Real-Time Age Prediction**  
  Uses a VGG-style CNN to predict your age within milliseconds.

- 🎯 **Face Detection with Margin**  
  Automatically detects and crops the largest face before inference using OpenCV Haar cascades.

- 🌐 **Cross-Origin Support**  
  FastAPI backend supports CORS for seamless frontend/backend integration.

- 🧪 **Custom Dataset Class**  
  Fully cleaned and filtered dataset loader to avoid noise, bad labels, and corrupt data.

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/KnowYourAge.git
cd KnowYourAge
```

### 2. Install Python Dependencies (Backend)

```bash
cd api
pip install -r requirements.txt
```

### 3. Start Backend API (FastAPI + Uvicorn)

```bash
uvicorn api.main:app --reload
```

### 4. Start Frontend (React)

```bash
cd frontend
npm install
npm start
```

Make sure `.env` contains:
```env
REACT_APP_API_URL=http://localhost:8000
```

---

## 📷 Sample Output

> "You look like you're **25**"  
(Displayed over your detected and cropped face image)

---

## 🧩 Tech Stack

| Layer      | Technology             |
|------------|------------------------|
| Frontend   | React + TailwindCSS    |
| Backend    | FastAPI + Uvicorn      |
| ML Model   | PyTorch (CNN DEX)      |
| Dataset    | UTKFace (cleaned)      |
| Face Detection | OpenCV (Haar cascade) |

---

## 👤 Author

**Ömer Yenal**  
> AI enthusiast building vision models from scratch  
> [LinkedIn](https://www.linkedin.com/omer-yenal) • [GitHub](https://github.com/omeryenal)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).