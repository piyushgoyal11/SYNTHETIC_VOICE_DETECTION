
import librosa
import numpy as np


def extract_features(audio_path):

    # Load audio and convert it to 16 kHz mono
    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    # Check audio length
    if len(y) < 1600:
        raise ValueError("Audio is too short.")

    # Normalize audio
    y = librosa.util.normalize(y)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    # Extract additional audio features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = []

    # Calculate mean and standard deviation of MFCC
    for feature in mfcc:
        features.append(np.mean(feature))
        features.append(np.std(feature))

    # Calculate additional features
    for feature in [
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing_rate,
        rms
    ]:
        features.append(np.mean(feature))
        features.append(np.std(feature))

    return np.array(features, dtype=np.float32)
