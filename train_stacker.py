import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from graph.nodes.inference import ml_inference
from graph.nodes.load_data import df_dev
from models.bert_model import get_active_bert_metadata
from models.fusion_features import build_signal_features, get_signal_feature_columns

training_data_path = "data/phishing_url_dataset_unique.csv -> url_sample -> df_dev"
training_label_column = "label"
results_output = "data/results/stacker_training_features.csv"
meta_model_dir = "data/ml_models/meta_model"
timezone = "US/Eastern"


def normalize_training_labels(df: pd.DataFrame, label_col: str, phishing_value: int) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = df[label_col].astype(int)

    if phishing_value == 1:
        df[label_col] = df[label_col].map({1: 0, 0: 1})
    elif phishing_value != 0:
        raise ValueError("Unsupported phishing label encoding")

    return df


def build_feature_dataset(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building feature rows"):
        state = ml_inference({"url": row.url})
        features = build_signal_features(state)
        features["url"] = row.url
        features["label"] = int(getattr(row, label_column))
        rows.append(features)

    return pd.DataFrame(rows)


def build_training_pipeline():
    base_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    return CalibratedClassifierCV(base_pipeline, method="sigmoid", cv=5)


def evaluate_stacker_cv(df_features: pd.DataFrame, label_column: str = "label") -> dict:
    feature_columns = get_signal_feature_columns()
    X = df_features[feature_columns]
    y = df_features[label_column].astype(int)

    estimator = build_training_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    probas = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
    benign_scores = probas[:, 1]
    y_pred = (benign_scores >= 0.5).astype(int)

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, benign_scores),
        "Num Samples": len(df_features),
    }


def train_stacker_model(df_features: pd.DataFrame, label_column: str = "label"):
    feature_columns = get_signal_feature_columns()
    X = df_features[feature_columns]
    y = df_features[label_column].astype(int)

    model = build_training_pipeline()
    model.fit(X, y)
    return model, feature_columns


def save_stacker_artifacts(model, feature_columns: list[str], metadata: dict):
    os.makedirs(meta_model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(meta_model_dir, "logistic_regression_calibrated.pkl"))
    joblib.dump(feature_columns, os.path.join(meta_model_dir, "signal_feature_columns.pkl"))

    with open(os.path.join(meta_model_dir, "meta_model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    print("loading training split")
    
    print("building features")
    df_features = build_feature_dataset(df_dev, label_column=training_label_column)
    df_features.to_csv(results_output, index=False)

    print("running CV")
    cv_metrics = evaluate_stacker_cv(df_features)

    print("fitting final calibrated model")
    model, feature_columns = train_stacker_model(df_features)

    bert_metadata = get_active_bert_metadata()
    metadata = {
        "saved_at": datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S"),
        "training_data_path": training_data_path,
        "label_column": training_label_column,
        "feature_columns": feature_columns,
        "bert_architecture": bert_metadata["bert_architecture"],
        "calibration_method": "sigmoid",
        "label_convention": {
            "benign": 1,
            "phishing": 0,
        },
        "cv_metrics": cv_metrics,
    }

    print("saving artifacts")
    save_stacker_artifacts(model, feature_columns, metadata)

    print("Stacker training complete.")
    print(cv_metrics)


if __name__ == "__main__":
    main()
