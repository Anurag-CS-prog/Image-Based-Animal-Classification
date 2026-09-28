import tensorflow as tf
import numpy as np

from src.preprocessing.preprocess import preprocess_image
from src.utils.config import MODEL_PATH


def predict_animal(image_path):
    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Preprocess image
    image = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(image, verbose=0)[0][0]

    # Convert prediction into class
    if prediction >= 0.5:
        label = "Cattle"
        confidence = prediction
    else:
        label = "Buffalo"
        confidence = 1 - prediction

    return label, confidence


if __name__ == "__main__":
    image_path = input("Enter image path: ")

    label, confidence = predict_animal(image_path)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")