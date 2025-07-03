import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Path to your EEG model (use .keras if available)
MODEL_PATH = "models/eeg_model.h5"

# Load the EEG model
def load_eeg_model():
    return load_model(MODEL_PATH, compile=False)

# Extract signal from EEG graph image
def extract_signal_from_image(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("L"))  # Convert to grayscale
    edges = cv2.Canny(img, 50, 150)     # Edge detection
    signal = np.mean(edges, axis=1) / 255.0  # Normalize signal

    # Resize to 178 features to match training input
    signal_resized = cv2.resize(signal.reshape(-1, 1), (1, 178), interpolation=cv2.INTER_LINEAR).flatten()
    return signal_resized

# Predict seizure from signal
def predict_from_signal(signal: np.ndarray) -> int:
    model = load_eeg_model()
    signal = signal.reshape(1, -1)  # Shape: (1, 178)
    prediction = model.predict(signal)[0][0]
    return int(prediction > 0.5)
