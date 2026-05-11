import json
from pathlib import Path
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/medical_image_model.keras")
CLASS_PATH = Path("models/medical_image_classes.json")

IMG_SIZE = (160, 160)

_model = None
_class_names = None


def load_model_once():
    global _model, _class_names

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _class_names is None:
        if not CLASS_PATH.exists():
            raise FileNotFoundError(f"Class file not found: {CLASS_PATH}")
        with open(CLASS_PATH, "r") as f:
            _class_names = json.load(f)

    return _model, _class_names


def preprocess_image(uploaded_file):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def analyze_medical_image(uploaded_file):
    model, class_names = load_model_once()

    processed_image = preprocess_image(uploaded_file)

    predictions = model(processed_image, training=False).numpy()[0]

    prediction_results = []
    for class_name, confidence in zip(class_names, predictions):
        prediction_results.append(
            {
                "class": class_name,
                "confidence": round(float(confidence) * 100, 2),
            }
        )

    prediction_results = sorted(
        prediction_results,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    top_prediction = prediction_results[0]

    return {
        "top_prediction": top_prediction["class"],
        "confidence": top_prediction["confidence"],
        "all_predictions": prediction_results,
        "description": (
            f"AI model predicts {top_prediction['class']} "
            f"with {top_prediction['confidence']}% confidence."
        ),
    }