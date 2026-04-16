from pathlib import Path

import joblib
import pandas as pd

meta_model_dirs = {
    "rich": Path("data/ml_models/meta_model_v2"),
    "4signal": Path("data/ml_models/meta_model_4signal_v1"),
}

_meta_models = {}
_meta_feature_columns = {}


def load_meta_model(stacker_variant: str = "rich"):
    if stacker_variant not in meta_model_dirs:
        raise ValueError(f"Unsupported stacker_variant: {stacker_variant}")

    if stacker_variant not in _meta_models:
        meta_model_path = meta_model_dirs[stacker_variant] / "logistic_regression_calibrated.pkl"
        _meta_models[stacker_variant] = joblib.load(meta_model_path)
    return _meta_models[stacker_variant]


def load_meta_feature_columns(stacker_variant: str = "rich"):
    if stacker_variant not in meta_model_dirs:
        raise ValueError(f"Unsupported stacker_variant: {stacker_variant}")

    if stacker_variant not in _meta_feature_columns:
        meta_feature_columns_path = meta_model_dirs[stacker_variant] / "signal_feature_columns.pkl"
        _meta_feature_columns[stacker_variant] = joblib.load(meta_feature_columns_path)
    return _meta_feature_columns[stacker_variant]


def predict_meta_model(features: dict, stacker_variant: str = "rich") -> dict:
    model = load_meta_model(stacker_variant=stacker_variant)
    feature_columns = load_meta_feature_columns(stacker_variant=stacker_variant)

    X = pd.DataFrame([[features.get(col) for col in feature_columns]], columns=feature_columns)
    proba = model.predict_proba(X)[0]

    phishing_prob = float(proba[0])
    benign_prob = float(proba[1])

    return {
        "stacking_phishing_prob": phishing_prob,
        "stacking_score": benign_prob,
        "stacking_prediction": "Benign" if benign_prob >= 0.5 else "Phishing",
    }
