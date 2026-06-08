from pathlib import Path

import joblib
import pandas as pd

meta_model_dirs = {
    "rich": Path("data/ml_models/meta_model_v2"),
    "4signal": Path("data/ml_models/meta_model_4signal_v1"),
}

_meta_models = {}
_meta_feature_columns = {}


def get_meta_model_dir(stacker_variant: str = "rich", model_dir: str | Path | None = None) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    if stacker_variant not in meta_model_dirs:
        raise ValueError(f"Unsupported stacker_variant: {stacker_variant}")
    return meta_model_dirs[stacker_variant]


def get_meta_model_name(stacker_variant: str = "rich", model_dir: str | Path | None = None) -> str:
    model_root = get_meta_model_dir(stacker_variant=stacker_variant, model_dir=model_dir)
    return model_root.name


def _get_model_cache_key(stacker_variant: str, model_dir: str | Path | None) -> str:
    resolved_dir = get_meta_model_dir(stacker_variant=stacker_variant, model_dir=model_dir)
    return str(resolved_dir.resolve())


def _find_model_artifact(model_root: Path) -> Path:
    for filename in ["meta_model.pkl", "logistic_regression_calibrated.pkl"]:
        candidate = model_root / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported meta-model artifact found under {model_root}")


def load_meta_model(stacker_variant: str = "rich", model_dir: str | Path | None = None):
    cache_key = _get_model_cache_key(stacker_variant=stacker_variant, model_dir=model_dir)
    if cache_key not in _meta_models:
        model_root = get_meta_model_dir(stacker_variant=stacker_variant, model_dir=model_dir)
        meta_model_path = _find_model_artifact(model_root)
        _meta_models[cache_key] = joblib.load(meta_model_path)
    return _meta_models[cache_key]


def load_meta_feature_columns(stacker_variant: str = "rich", model_dir: str | Path | None = None):
    cache_key = _get_model_cache_key(stacker_variant=stacker_variant, model_dir=model_dir)
    if cache_key not in _meta_feature_columns:
        model_root = get_meta_model_dir(stacker_variant=stacker_variant, model_dir=model_dir)
        meta_feature_columns_path = model_root / "signal_feature_columns.pkl"
        _meta_feature_columns[cache_key] = joblib.load(meta_feature_columns_path)
    return _meta_feature_columns[cache_key]


def predict_meta_model(
    features: dict,
    stacker_variant: str = "rich",
    model_dir: str | Path | None = None,
) -> dict:
    model = load_meta_model(stacker_variant=stacker_variant, model_dir=model_dir)
    feature_columns = load_meta_feature_columns(stacker_variant=stacker_variant, model_dir=model_dir)

    X = pd.DataFrame([[features.get(col) for col in feature_columns]], columns=feature_columns)
    proba = model.predict_proba(X)[0]

    phishing_prob = float(proba[0])
    benign_prob = float(proba[1])

    return {
        "stacking_phishing_prob": phishing_prob,
        "stacking_score": benign_prob,
        "stacking_prediction": "Benign" if benign_prob >= 0.5 else "Phishing",
    }
