import streamlit as st
import joblib
import librosa
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from datetime import datetime

from audio_utils import extract_features


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="VoiceGuard AI | Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "voice_detector.pkl"


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #0b1120;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: #111b2e;
    border-right: 1px solid #263449;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
}

p, label {
    color: #cbd5e1;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    color: #60a5fa;
    letter-spacing: 1px;
}

.brand-sub {
    font-size: 11px;
    color: #94a3b8;
    letter-spacing: 2px;
}

.hero {
    background: linear-gradient(120deg, #172554, #172033);
    border: 1px solid #294477;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
}

.hero-text {
    color: #bfdbfe;
    font-size: 15px;
}

.metric-card {
    background: #131f33;
    border: 1px solid #293b55;
    border-radius: 14px;
    padding: 20px;
    min-height: 120px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.panel {
    background: #131f33;
    border: 1px solid #293b55;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 18px;
}

.result-real {
    background: #102c25;
    border: 1px solid #16845c;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

.result-spoof {
    background: #351b25;
    border: 1px solid #b44b62;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

.result-title {
    font-size: 30px;
    font-weight: 900;
    color: white;
}

.small-muted {
    color: #94a3b8;
    font-size: 12px;
}

div.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 9px;
    height: 48px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
    border: none;
}

hr {
    border-color: #293b55;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Trained model not found. Check models/voice_detector.pkl.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ==========================================
# SESSION HISTORY
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        '<div class="brand">🎙️ VOICEGUARD AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-sub">SYNTHETIC VOICE INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Voice Detection", "History", "About"],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")
    st.success("AI Model Loaded")

    st.markdown(
        '<p class="small-muted">Random Forest Classifier</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<p class="small-muted">P72 PROJECT<br>'
        'Synthetic Voice Detection System</p>',
        unsafe_allow_html=True
    )


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">AI Voice Detection Dashboard</div>
        <div class="hero-text">
            Analyse audio recordings and identify potentially
            synthetic or manipulated speech.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## System Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MODEL TYPE</div>
            <div class="metric-value">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">TRAINING RECORDINGS</div>
            <div class="metric-value">25,380</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">HELD-OUT SPLIT ACCURACY</div>
            <div class="metric-value">97.40%</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Accuracy is from the random held-out split of the training "
        "dataset, not an independent real-world evaluation."
    )

    st.write("")

    st.markdown("## Quick Actions")

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="panel">
            <h3>🎧 Voice Detection</h3>
            <p>Upload an audio file and analyse its authenticity.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Voice Detection"):
            st.session_state.page_choice = "Voice Detection"
            st.rerun()

    with right:
        st.markdown("""
        <div class="panel">
            <h3>📊 Detection History</h3>
            <p>Review predictions made during this session.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open History"):
            st.session_state.page_choice = "History"
            st.rerun()

    st.markdown("## Model Information")

    st.info(
        "The current system uses MFCC and spectral audio features "
        "with a Random Forest classifier. Predictions are model "
        "estimates and should be independently validated."
    )


# ==========================================
# VOICE DETECTION PAGE
# ==========================================

elif page == "Voice Detection":

    st.markdown("## 🎧 Voice Detection")

    st.write(
        "Upload a WAV or FLAC recording to analyse whether "
        "the voice is genuine or synthetic."
    )

    uploaded_file = st.file_uploader(
        "Upload audio recording",
        type=["wav", "flac"],
        help="Choose a WAV or FLAC audio recording."
    )

    if uploaded_file is not None:

        st.success(f"File uploaded: {uploaded_file.name}")

        st.audio(uploaded_file)

        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)

        st.caption(
            f"File size: {file_size:.2f} MB | "
            f"Format: {Path(uploaded_file.name).suffix.upper()}"
        )

        if st.button("ANALYSE VOICE"):

            temp_path = None

            try:
                with st.spinner(
                    "Extracting audio features and analysing voice..."
                ):

                    suffix = Path(uploaded_file.name).suffix

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp_file:

                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name

                    features = extract_features(temp_path)

                    prediction = model.predict([features])[0]

                    label = str(prediction).strip().lower()

                    if label in ["bonafide", "bona fide", "genuine"]:
                        result = "BONAFIDE"

                    elif label in ["spoof", "synthetic"]:
                        result = "SPOOF"

                    else:
                        result = str(prediction)

                    probabilities = None
                    confidence = None

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(
                            [features]
                        )[0]

                        classes = list(model.classes_)

                        predicted_index = classes.index(prediction)

                        confidence = float(
                            probabilities[predicted_index] * 100
                        )

                st.markdown("### Analysis Result")

                if result == "BONAFIDE":

                    st.markdown("""
                    <div class="result-real">
                        <div class="result-title">✓ BONAFIDE</div>
                        <p>Genuine voice predicted by the model</p>
                    </div>
                    """, unsafe_allow_html=True)

                elif result == "SPOOF":

                    st.markdown("""
                    <div class="result-spoof">
                        <div class="result-title">⚠ SPOOF</div>
                        <p>Synthetic or manipulated voice predicted</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.warning(f"Model returned an unknown label: {result}")

                if confidence is not None:
                    st.write("")
                    st.markdown("#### Prediction confidence")
                    st.progress(confidence / 100)
                    st.write(f"**{confidence:.2f}%**")

                st.caption(
                    "Model confidence is not a guarantee of correctness."
                )

                # Audio waveform
                st.markdown("### Audio Waveform")

                y, sr = librosa.load(
                    temp_path,
                    sr=16000,
                    mono=True
                )

                waveform_data = y[::max(1, len(y) // 1500)]

                st.line_chart(
                    pd.DataFrame({"Amplitude": waveform_data}),
                    height=200
                )

                duration = len(y) / sr

                st.markdown("### Audio Information")

                a, b, c = st.columns(3)

                a.metric("Duration", f"{duration:.2f} sec")
                b.metric("Sample Rate", f"{sr} Hz")
                c.metric("File Size", f"{file_size:.2f} MB")

                st.session_state.history.insert(
                    0,
                    {
                        "File": uploaded_file.name,
                        "Prediction": result,
                        "Confidence": (
                            f"{confidence:.2f}%"
                            if confidence is not None
                            else "Unavailable"
                        ),
                        "Time": datetime.now().strftime("%H:%M:%S")
                    }
                )

            except Exception as error:
                st.error(f"Analysis failed: {error}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# ==========================================
# HISTORY PAGE
# ==========================================

elif page == "History":

    st.markdown("## 📊 Detection History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info(
            "No recordings analysed yet. Visit Voice Detection "
            "to start your first analysis."
        )


# ==========================================
# ABOUT PAGE
# ==========================================

elif page == "About":

    st.markdown("## About VoiceGuard AI")

    st.write(
        "VoiceGuard AI is a machine-learning-based synthetic "
        "voice detection system developed as a P72 project."
    )

    st.markdown("### Technology Stack")

    st.markdown("""
    - **Python:** Main programming language
    - **Streamlit:** Frontend and web interface
    - **Librosa:** Audio processing and feature extraction
    - **NumPy:** Numerical computation
    - **Scikit-learn:** Random Forest classifier
    - **Joblib:** Saving and loading the trained model
    - **ASVspoof 2019 LA:** Training dataset
    """)

    st.markdown("### Detection Categories")

    st.success("BONAFIDE — Genuine speech recording")
    st.error("SPOOF — Spoofed or synthetic speech recording")

    st.warning(
        "This is an experimental detection system. "
        "Its predictions are not guaranteed to be correct."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center><span style='color:#94a3b8'>"
    "VOICEGUARD AI | P72 PROJECT | Synthetic Voice Detection System"
    "</span></center>",
    unsafe_allow_html=True
)