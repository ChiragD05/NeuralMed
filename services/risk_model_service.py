import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/risk_model.pkl")
META_PATH = Path("models/risk_model_meta.json")

_model = None
_meta = None


def load_once():
    global _model, _meta

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Risk model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)

    if _meta is None:
        if not META_PATH.exists():
            raise FileNotFoundError(f"Risk metadata not found: {META_PATH}")
        with open(META_PATH, "r") as f:
            _meta = json.load(f)

    return _model, _meta


def predict_risk(input_data: dict):
    model, meta = load_once()

    df = pd.DataFrame([input_data])
    risk_prob = float(model.predict_proba(df)[0][1])
    risk_label = "High Risk" if risk_prob >= 0.5 else "Low Risk"

    return {
        "risk_probability": round(risk_prob * 100, 2),
        "risk_label": risk_label,
        "raw_probability": risk_prob,
        "meta": meta,
    }