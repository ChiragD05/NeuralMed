import json
from pathlib import Path

import joblib

MODEL_PATH = Path("models/symptom_model.pkl")
CLASSES_PATH = Path("models/symptom_classes.json")

_model = None
_class_names = None


def load_model_once():
    global _model, _class_names

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)

    if _class_names is None:
        if not CLASSES_PATH.exists():
            raise FileNotFoundError(f"Class file not found: {CLASSES_PATH}")
        with open(CLASSES_PATH, "r") as f:
            _class_names = json.load(f)

    return _model, _class_names


def predict_symptom_disease(symptom_text: str, top_k: int = 3):
    model, class_names = load_model_once()

    probs = model.predict_proba([symptom_text])[0]
    ranked = sorted(
        [
            {"condition": cls, "confidence": round(float(prob) * 100, 2)}
            for cls, prob in zip(class_names, probs)
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "top_prediction": ranked[0]["condition"],
        "confidence": ranked[0]["confidence"],
        "all_predictions": ranked[:top_k],
    }