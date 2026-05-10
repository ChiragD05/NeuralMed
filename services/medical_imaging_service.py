import random
import numpy as np
from PIL import Image

MEDICAL_CLASSES = [
    "Normal",
    "Pneumonia",
    "COVID-19",
    "Tuberculosis",
]

CLASS_DESCRIPTIONS = {
    "Normal": "No strong abnormality pattern detected.",
    "Pneumonia": "Lung opacity patterns may indicate pneumonia.",
    "COVID-19": "Potential COVID-related lung involvement detected.",
    "Tuberculosis": "Possible tuberculosis-related findings detected.",
}

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    return image

def fake_cnn_prediction():
    """
    Simulated CNN probabilities.
    Replace later with real model inference.
    """

    probs = np.random.dirichlet(np.ones(len(MEDICAL_CLASSES)), size=1)[0]

    predictions = []

    for cls, prob in zip(MEDICAL_CLASSES, probs):
        predictions.append({
            "class": cls,
            "confidence": round(float(prob) * 100, 2)
        })

    predictions = sorted(
        predictions,
        key=lambda x: x["confidence"],
        reverse=True
    )

    top_prediction = predictions[0]

    return {
        "top_prediction": top_prediction["class"],
        "confidence": top_prediction["confidence"],
        "description": CLASS_DESCRIPTIONS[top_prediction["class"]],
        "all_predictions": predictions
    }

def analyze_medical_image(uploaded_file):
    image = preprocess_image(uploaded_file)

    prediction_result = fake_cnn_prediction()

    return prediction_result