from pathlib import Path

import joblib
import pandas as pd

meta_model_dir = Path("data/ml_models/meta_model_v2")
meta_model_path = meta_model_dir / "logistic_regression_calibrated.pkl"
meta_feature_columns_path = meta_model_dir / "signal_feature_columns.pkl"

_meta_model = None
_meta_feature_columns = None


def load_meta_model():
    global _meta_model
    if _meta_model is None:
        _meta_model = joblib.load(meta_model_path)
    return _meta_model


def load_meta_feature_columns():
    global _meta_feature_columns
    if _meta_feature_columns is None:
        _meta_feature_columns = joblib.load(meta_feature_columns_path)
    return _meta_feature_columns


def predict_meta_model(features: dict) -> dict:
    model = load_meta_model()
    feature_columns = load_meta_feature_columns()

    X = pd.DataFrame([[features.get(col) for col in feature_columns]], columns=feature_columns)
    proba = model.predict_proba(X)[0]

    phishing_prob = float(proba[0])
    benign_prob = float(proba[1])

    return {
        "stacking_phishing_prob": phishing_prob,
        "stacking_score": benign_prob,
        "stacking_prediction": "Benign" if benign_prob >= 0.5 else "Phishing",
    }
