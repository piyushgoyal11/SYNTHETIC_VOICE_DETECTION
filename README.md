# 🎙️ VoiceGuard AI

### AI-Powered Synthetic Voice & Deepfake Audio Detection

\<p align="center">   \*\*Detect. Analyze. Protect.\*\* \</p>

VoiceGuard AI is a machine learning-based application designed to distinguish genuine human speech from synthetic or spoofed audio recordings. It uses a trained Random Forest classifier and an interactive Streamlit dashboard to analyze uploaded audio files.

---

## 🌟 Project Highlights

- 🎧 **Audio Analysis:** Upload audio recordings for authenticity analysis.
- 🤖 **Machine Learning:** Uses a trained Random Forest classifier.
- 🔍 **Real vs Spoof Detection:** Classifies audio into two categories.
- 📊 **Interactive Dashboard:** Modern, dark-themed Streamlit interface.
- 📜 **Detection History:** Review predictions made during the current session.
- 📈 **Model Performance:** Displays training information and evaluation accuracy.

---

## 🖥️ Application Preview

**Live Application:**
[🚀 Launch VoiceGuard AI](https://syntheticvoicedetection-69cbmb3bppz9cz2f59au.streamlit.app/)

---

## 🧠 Machine Learning Model

VoiceGuard AI uses a Random Forest classification algorithm trained on real and spoof audio recordings.

| Parameter                  | Details                  |
| -------------------------- | ------------------------ |
| Algorithm                  | Random Forest Classifier |
| Training Recordings        | 25,380                   |
| Reported Held-out Accuracy | 97.40%                   |
| Classification Type        | Binary Classification    |
| Output Classes             | Real / Spoof             |

**Note:** The reported accuracy is from a held-out dataset split. It does not represent independently verified real-world performance.

---

## ⚙️ Technology Stack

| Technology   | Purpose                       |
| ------------ | ----------------------------- |
| Python       | Core programming language     |
| Streamlit    | Web application and dashboard |
| Scikit-learn | Machine learning              |
| Librosa      | Audio processing              |
| NumPy        | Numerical computation         |
| Pandas       | Data handling                 |
| SoundFile    | Audio file processing         |
| Joblib       | Model serialization           |

---

## 📂 Project Structure

```text
SYNTHETIC_VOICE_DETECTION/
│
├── app.py
├── train.py
├── audio_utils.py
├── requirements.txt
├── README.md
│
├── models/
│   └── voice_detector.pkl
│
└── raw_data/
    └── train/
        └── flac/
```

*The raw audio dataset is stored separately and may not be included in the GitHub repository.*

---

## 🚀 Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SYNTHETIC_VOICE_DETECTION.git
```

### Step 2: Navigate to the Project

```bash
cd SYNTHETIC_VOICE_DETECTION
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment (Windows)

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 🔬 How It Works

1. **Upload Audio:** The user uploads a supported audio recording.
2. **Audio Processing:** The application processes the recording and extracts the features required by the model.
3. **AI Classification:** The trained Random Forest model analyzes the extracted features.
4. **Prediction:** The system returns a Real or Spoof classification.
5. **History:** The prediction can be reviewed in the dashboard's history section.

---

## 🎯 Project Objective

The objective of VoiceGuard AI is to explore the use of machine learning in identifying synthetic and manipulated speech.

The system demonstrates how audio processing and machine learning can contribute to research into voice authenticity, audio security, and deepfake detection.

---

## ⚠️ Limitations

- Prediction quality depends on audio quality and the training data.
- Unseen voice-generation methods may produce incorrect predictions.
- A classification result should not be treated as definitive proof of authenticity.
- The reported dataset accuracy may differ from real-world performance.

---

## 🌐 Live Deployment

**Platform:** Streamlit Community Cloud

🔗 [Open VoiceGuard AI](https://syntheticvoicedetection-69cbmb3bppz9cz2f59au.streamlit.app/)

---

## 👨‍💻 Project Information

| Field     | Details                                    |
| --------- | ------------------------------------------ |
| Project   | VoiceGuard AI                              |
| Domain    | Artificial Intelligence & Machine Learning |
| Category  | Synthetic Voice Detection                  |
| Model     | Random Forest                              |
| Interface | Streamlit                                  |
| Language  | Python                                     |

---

## ⭐ Acknowledgements

This project was developed as an exploration of machine learning applications in synthetic voice detection and audio authenticity analysis.

If you find this project interesting, consider giving the repository a ⭐ star!

**VoiceGuard AI — Detecting Synthetic Voices with AI.**
