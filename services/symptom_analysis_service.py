import re
from collections import Counter

# Disease patterns
SYMPTOM_DATABASE = {
    "Common Cold": [
        "sneezing",
        "runny nose",
        "sore throat",
        "mild cough",
    ],

    "Flu": [
        "fever",
        "body ache",
        "fatigue",
        "chills",
        "headache",
    ],

    "COVID-19": [
        "fever",
        "dry cough",
        "loss of taste",
        "loss of smell",
        "fatigue",
    ],

    "Migraine": [
        "headache",
        "nausea",
        "light sensitivity",
        "vomiting",
    ],

    "Pneumonia": [
        "chest pain",
        "shortness of breath",
        "cough",
        "fever",
    ],

    "Gastritis": [
        "stomach pain",
        "bloating",
        "vomiting",
        "acidity",
    ],
}

EMERGENCY_KEYWORDS = [
    "chest pain",
    "severe shortness of breath",
    "unconscious",
    "seizure",
    "stroke",
    "heavy bleeding",
]

STOPWORDS = {
    "i",
    "have",
    "am",
    "is",
    "the",
    "a",
    "an",
    "and",
    "with",
    "my",
    "feeling",
}

def preprocess_text(text: str):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.split()

    tokens = [t for t in tokens if t not in STOPWORDS]

    return tokens

def detect_emergency(text: str):

    text = text.lower()

    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text:
            return True

    return False

def calculate_match_score(user_tokens, disease_symptoms):

    matched = []

    for symptom in disease_symptoms:

        symptom_tokens = symptom.lower().split()

        if any(token in user_tokens for token in symptom_tokens):
            matched.append(symptom)

    score = len(matched) / len(disease_symptoms)

    return score, matched

def analyze_symptoms(user_input: str):

    user_tokens = preprocess_text(user_input)

    emergency = detect_emergency(user_input)

    predictions = []

    for disease, symptoms in SYMPTOM_DATABASE.items():

        score, matched = calculate_match_score(
            user_tokens,
            symptoms
        )

        if score > 0:

            predictions.append({
                "condition": disease,
                "confidence": round(score * 100, 2),
                "matched_symptoms": matched,
            })

    predictions = sorted(
        predictions,
        key=lambda x: x["confidence"],
        reverse=True
    )

    if emergency:
        severity = "High"
        recommendation = (
            "Emergency symptoms detected. Seek immediate medical attention."
        )

    elif predictions and predictions[0]["confidence"] > 70:
        severity = "Moderate to High"
        recommendation = (
            "Consult a healthcare professional for further evaluation."
        )

    elif predictions:
        severity = "Moderate"
        recommendation = (
            "Monitor symptoms and consult a doctor if symptoms worsen."
        )

    else:
        severity = "Low"
        recommendation = (
            "No strong disease pattern detected."
        )

    return {
        "severity": severity,
        "emergency": emergency,
        "predictions": predictions[:3],
        "recommendation": recommendation,
    }