
import os
import glob
import joblib
import numpy as np

from audio_utils import extract_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Dataset folders
DATASET_PATH = "dataset/train"
MODEL_PATH = "models/voice_detector.pkl"

X = []
y = []

# Read genuine and spoof audio files
for label in ["bonafide", "spoof"]:

    folder = os.path.join(DATASET_PATH, label)

    files = (
    glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)
    + glob.glob(os.path.join(folder, "**", "*.flac"), recursive=True)
)

    print(f"{label}: {len(files)} audio files found")

    for file in files:
        try:
            features = extract_features(file)
            X.append(features)
            y.append(label)

        except Exception as error:
            print(f"Skipping {file}: {error}")

# Check dataset
if len(X) < 20:
    raise ValueError(
        "Not enough usable audio files. "
        "Please add at least 20 labelled WAV recordings."
    )

if len(set(y)) != 2:
    raise ValueError(
        "Both bonafide and spoof recordings are required."
    )

X = np.array(X)
y = np.array(y)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train model
print("\nTraining started...")
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)