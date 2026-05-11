import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/risk_prediction/framingham.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "risk_model.pkl"
META_PATH = MODEL_DIR / "risk_model_meta.json"

TARGET_COL = "TenYearCHD"

FEATURES = [
    "male",
    "age",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Keep only required columns
missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns in dataset: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

df = df[FEATURES + [TARGET_COL]].copy()

# Basic cleanup
df = df.dropna(subset=[TARGET_COL])

X = df[FEATURES]
y = df[TARGET_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

numeric_features = FEATURES

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )),
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {auc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_PATH)

meta = {
    "target": TARGET_COL,
    "features": FEATURES,
    "accuracy": float(acc),
    "roc_auc": float(auc),
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved metadata to: {META_PATH}")