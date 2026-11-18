# 🎙️ VoiceGuard AI – Synthetic Voice Detection System

**An AI-powered system for detecting synthetic, spoofed, and manipulated voice recordings.**

## 📌 Project Overview

VoiceGuard AI is a machine learning-based application designed to distinguish between genuine human voice recordings and synthetic or spoofed audio.

The project uses a Random Forest classifier trained on real and spoof audio recordings. It provides an interactive Streamlit dashboard where users can upload audio files and analyze their authenticity.

## ✨ Key Features

- 🎧 **Audio Upload:** Upload audio recordings for analysis.
- 🤖 **AI-Powered Detection:** Uses a trained Random Forest classification model.
- 🔍 **Real vs Spoof Classification:** Identifies potentially genuine and synthetic voice recordings.
- 📊 **Interactive Dashboard:** Professional Streamlit-based interface.
- 📜 **Detection History:** Review predictions made during the current session.
- 📈 **Model Performance:** Displays training information and held-out split accuracy.

## 🧠 Machine Learning Model

| Component | Details |
|---|---|
| Model | Random Forest Classifier |
| Training Recordings | 25,380 |
| Held-out Split Accuracy | 97.40% |
| Task | Binary Audio Classification |
| Output | Real / Spoof |

**Note:** The reported accuracy is based on a held-out dataset split and does not guarantee equivalent performance on unseen real-world audio.

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Librosa
- SoundFile
- Joblib

## 📂 Project Structure

```text
SYNTHETIC_VOICE_DETECTION/
│
├── app.py
├── train.py
├── audio_utils.py
├── requirements.txt
│
├── models/
│   └── voice_detector.pkl
│
├── raw_data/
│   └── train/
│       └── flac/
│
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/piyushgoyal11/SYNTHETIC_VOICE_DETECTION.git
```

### 2. Open the project folder

```bash
cd SYNTHETIC_VOICE_DETECTION
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🌐 Live Deployment

**VoiceGuard AI is deployed using Streamlit Community Cloud.**

[Open Live Application](https://syntheticvoicedetection-69cbmb3bppz9cz2f59au.streamlit.app/)

## 📊 Model Performance

The trained Random Forest classifier achieved a reported held-out split accuracy of **97.40%** on the training evaluation split.

Actual performance may vary depending on audio quality, recording devices, compression, and previously unseen spoofing techniques.

## 🎯 Project Objective

The main objective of VoiceGuard AI is to explore how machine learning can help identify synthetic and manipulated speech, supporting research into audio authenticity and voice-based security.

## 👨‍💻 Project Information

**Project Name:** VoiceGuard AI  
**Project Type:** Machine Learning / Artificial Intelligence  
**Domain:** Synthetic Voice Detection  
**Interface:** Streamlit Dashboard  
**Model:** Random Forest Classifier

---

⭐ If you find this project interesting, consider giving the repository a star!

**Developed as an AI/ML project focused on synthetic voice detection.**
