# src/model_utils.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Mapping folder names to simplified class labels
label_map = {
    'Mild Impairment': 'Mild',
    'Moderate Impairment': 'Moderate',
    'No Impairment': 'No',
    'Very Mild Impairment': 'VeryMild'
}
labels = list(label_map.values())

# Load Alzheimer's MRI model
def load_model():
    return tf.keras.models.load_model("models/best_model.keras", compile=False)

# Predict the Alzheimer's stage from MRI image
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return labels[predicted_class]
