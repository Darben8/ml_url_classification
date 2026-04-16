import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

feature_source_path = "data/results/stacker_training_features.csv"
results_output = "data/results/stacker_training_features_4signal.csv"
train_results_output = "data/results/all_model_train_results.csv"
meta_model_dir = "data/ml_models/meta_model_4signal_v1"
timezone = "US/Eastern"


def get_4signal_feature_columns() -> list[str]:
    return ["bert_score", "cb_score", "vt_score", "tranco_score"]


def build_4signal_feature_dataset(source_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(source_path)

    df_out = pd.DataFrame(
        {
            "bert_score": df["bert_score"],
            "cb_score": df["cb_benign_prob"],
            "vt_score": 1 - df["vt_detection_rate"],
            "tranco_score": df["tranco_score"],
            "url": df["url"],
            "label": df["label"],
        }
    )
    df_out.loc[df["vt_detection_rate"].isna(), "vt_score"] = float("nan")
    df_out.to_csv(output_path, index=False)
    return df_out


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
    feature_columns = get_4signal_feature_columns()
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
    feature_columns = get_4signal_feature_columns()
    X = df_features[feature_columns]
    y = df_features[label_column].astype(int)

    model = build_training_pipeline()
    model.fit(X, y)
    return model, feature_columns


def measure_inference_time(model, df_features: pd.DataFrame, feature_columns: list[str]) -> float:
    X = df_features[feature_columns]
    start = time.time()
    model.predict_proba(X)
    return round(time.time() - start, 3)


def save_stacker_artifacts(model, feature_columns: list[str], metadata: dict):
    os.makedirs(meta_model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(meta_model_dir, "logistic_regression_calibrated.pkl"))
    joblib.dump(feature_columns, os.path.join(meta_model_dir, "signal_feature_columns.pkl"))
    with open(os.path.join(meta_model_dir, "meta_model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def append_train_results(
    cv_metrics: dict,
    saved_at: str,
    train_time_seconds: float,
    inference_time_seconds: float,
    num_samples: int,
):
    column_order = [
        "Model",
        "Accuracy",
        "CV Accuracy",
        "CV Std",
        "Precision",
        "Recall",
        "F1-Score",
        "Train Time (s)",
        "Inference Time (s)",
        "Saved_at",
        "Training dataset name",
        "ROC-AUC",
        "Num samples in dataset",
        "Note",
    ]

    row = {
        "Model": "Stacker model (4-signal LR meta model v1)",
        "Accuracy": cv_metrics.get("Accuracy"),
        "CV Accuracy": "",
        "CV Std": "",
        "Precision": cv_metrics.get("Precision"),
        "Recall": cv_metrics.get("Recall"),
        "F1-Score": cv_metrics.get("F1"),
        "Train Time (s)": round(train_time_seconds, 3),
        "Inference Time (s)": inference_time_seconds,
        "Saved_at": saved_at.replace(":", "-"),
        "Training dataset name": feature_source_path,
        "ROC-AUC": cv_metrics.get("ROC_AUC"),
        "Num samples in dataset": num_samples,
        "Note": f"sklearn v{sklearn.__version__}; 4-signal stacker",
    }

    df_out = pd.DataFrame([[row.get(col) for col in column_order]], columns=column_order)
    try:
        df_existing = pd.read_csv(train_results_output)
        if list(df_existing.columns) != column_order:
            print("Error: all_model_train_results.csv schema does not match expected columns.")
            print("Skipping append to all_model_train_results.csv.")
            return
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass

    df_out.to_csv(train_results_output, index=False)
    print(f"appended 4-signal stacker training summary to {train_results_output}")


def main():
    train_start = time.time()
    print("loading derived rich-signal feature file")
    print("building 4-signal feature dataset")
    df_features = build_4signal_feature_dataset(feature_source_path, results_output)

    print("running CV")
    cv_metrics = evaluate_stacker_cv(df_features)

    print("fitting final calibrated model")
    model, feature_columns = train_stacker_model(df_features)
    inference_time_seconds = measure_inference_time(model, df_features, feature_columns)

    saved_at = datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "saved_at": saved_at,
        "model_family": "4-signal",
        "feature_source_path": feature_source_path,
        "derived_feature_csv_path": results_output,
        "feature_columns": feature_columns,
        "calibration_method": "sigmoid",
        "label_convention": {
            "benign": 1,
            "phishing": 0,
        },
        "cv_metrics": cv_metrics,
        "note": "Derived from richer stacker_training_features.csv",
    }

    print("saving artifacts")
    save_stacker_artifacts(model, feature_columns, metadata)
    append_train_results(
        cv_metrics=cv_metrics,
        saved_at=saved_at,
        train_time_seconds=time.time() - train_start,
        inference_time_seconds=inference_time_seconds,
        num_samples=len(df_features),
    )

    print("4-signal stacker training complete.")
    print(cv_metrics)


if __name__ == "__main__":
    main()
