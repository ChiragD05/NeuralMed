import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/symptom_dataset/symptom2disease.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "symptom_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "symptom_vectorizer.pkl"
CLASSES_PATH = MODEL_DIR / "symptom_classes.json"

# -----------------------------
# Load dataset
# -----------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Columns in dataset:", list(df.columns))
print("First few rows:")
print(df.head())

# Try to detect text and label columns automatically
text_col_candidates = ["text", "symptoms", "Symptom", "Symptom_Description", "description", "content"]
label_col_candidates = ["label", "disease", "Disease", "target", "class"]

text_col = None
label_col = None

for col in text_col_candidates:
    if col in df.columns:
        text_col = col
        break

for col in label_col_candidates:
    if col in df.columns:
        label_col = col
        break

if text_col is None or label_col is None:
    raise ValueError(
        "Could not automatically detect text and label columns. "
        f"Available columns: {list(df.columns)}"
    )

df = df[[text_col, label_col]].dropna()
df[text_col] = df[text_col].astype(str)
df[label_col] = df[label_col].astype(str)

X = df[text_col]
y = df[label_col]

class_names = sorted(y.unique().tolist())
with open(CLASSES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)

print("Detected text column:", text_col)
print("Detected label column:", label_col)
print("Classes:", class_names)

# -----------------------------
# Train/validation split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() > 1 else None,
)

# -----------------------------
# Build model
# -----------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nValidation Accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, MODEL_PATH)

# Also save vectorizer separately if you want it later
joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)

print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved vectorizer to: {VECTORIZER_PATH}")
print(f"Saved classes to: {CLASSES_PATH}")