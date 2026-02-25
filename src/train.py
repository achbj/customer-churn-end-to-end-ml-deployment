import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODEL_DIR, PROCESSED_DIR
from src.preprocess import CATEGORICAL_COLS, NUMERIC_COLS


def load_splits():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    target = "churn"
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS

    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test


def build_preprocessor():
    """Column transformer that scales numeric and one-hot encodes categorical."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )


def get_models():
    """Return a dict of model name -> sklearn estimator."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])

    return {"accuracy": acc, "auc": auc, "report": report}


def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_splits()
    preprocessor = build_preprocessor()
    models = get_models()

    best_auc = 0
    best_name = ""
    best_pipeline = None

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)

        print(f"\n--- {name} ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC-ROC:  {metrics['auc']:.4f}")
        print(metrics["report"])

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_name = name
            best_pipeline = pipe

    # save the winner
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
    joblib.dump(best_pipeline, model_path)

    print(f"\nBest model: {best_name} (AUC = {best_auc:.4f})")
    print(f"Saved to {model_path}")

    return best_pipeline


if __name__ == "__main__":
    train_and_evaluate()
