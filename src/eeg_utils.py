import numpy as np              # For numerical operations
import cv2                      # OpenCV for image processing
from PIL import Image           # For image handling
from tensorflow.keras.models import load_model  # To load trained ML model


# Path to the trained EEG classification model
MODEL_PATH = "models/eeg_model.h5"


# -------------------------------
# Load the trained EEG model
# -------------------------------
def load_eeg_model():
    """
    Loads the pre-trained EEG deep learning model.
    compile=False because we only need prediction, not training.
    """
    return load_model(MODEL_PATH, compile=False)


# ---------------------------------------------
# Extract EEG signal values from EEG graph image
# ---------------------------------------------
def extract_signal_from_image(image: Image.Image) -> np.ndarray:
    """
    Converts an EEG graph image into a numerical signal
    that matches the model's training input format.
    """

    # Convert image to grayscale (1 channel instead of RGB)
    img = np.array(image.convert("L"))

    # Detect edges in the EEG waveform using Canny edge detection
    edges = cv2.Canny(img, 50, 150)

    # Convert edge image into a 1D signal by averaging each row
    # Divide by 255 to normalize values between 0 and 1
    signal = np.mean(edges, axis=1) / 255.0

    # Resize the signal to 178 values (same size used during training)
    signal_resized = cv2.resize(
        signal.reshape(-1, 1),     # Convert to column vector
        (1, 178),                  # Target shape
        interpolation=cv2.INTER_LINEAR
    ).flatten()

    return signal_resized


# ---------------------------------
# Predict seizure from EEG signal
# ---------------------------------
def predict_from_signal(signal: np.ndarray) -> int:
    """
    Takes a processed EEG signal and predicts
    whether a seizure is present or not.
    """

    # Load the trained model
    model = load_eeg_model()

    # Reshape signal to match model input: (batch_size, features)
    signal = signal.reshape(1, -1)

    # Get prediction probability from the model
    prediction = model.predict(signal)[0][0]

    # Convert probability to binary output (0 or 1)
    # > 0.5 → seizure detected
    return int(prediction > 0.5)
