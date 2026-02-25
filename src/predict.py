import os
from typing import Any

import joblib
import pandas as pd

from config import MODEL_DIR
from src.preprocess import CATEGORICAL_COLS, NUMERIC_COLS


def load_model(model_path: str | None = None):
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "churn_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python run_pipeline.py"
        )

    return joblib.load(model_path)


def predict_one(model, features: dict[str, Any]) -> dict[str, float]:
    """Return churn probability for a single record."""
    ordered = {}
    for col in NUMERIC_COLS:
        ordered[col] = float(features[col])
    for col in CATEGORICAL_COLS:
        ordered[col] = str(features[col])

    X = pd.DataFrame([ordered])
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    return {"churn_probability": proba, "churn_prediction": pred}
