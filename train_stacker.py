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
from tqdm import tqdm

from graph.nodes.inference import ml_inference
from graph.nodes.load_data import df_dev
from models.bert_model import get_active_bert_metadata
from models.fusion_features import build_signal_features, get_signal_feature_columns

training_data_path = "data/phishing_url_dataset_unique.csv -> url_sample -> df_dev"
training_label_column = "label"
results_output = "data/results/stacker_training_features.csv"
train_results_output = "data/results/all_model_train_results.csv"
meta_model_dir = "data/ml_models/meta_model_v2"
timezone = "US/Eastern"
batch_size = 50


def normalize_training_labels(df: pd.DataFrame, label_col: str, phishing_value: int) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = df[label_col].astype(int)

    if phishing_value == 1:
        df[label_col] = df[label_col].map({1: 0, 0: 1})
    elif phishing_value != 0:
        raise ValueError("Unsupported phishing label encoding")

    return df


def get_feature_output_columns() -> list[str]:
    return get_signal_feature_columns() + ["url", "label"]


def validate_existing_feature_csv(output_path: str):
    if not os.path.exists(output_path):
        return

    df_existing = pd.read_csv(output_path, nrows=0)
    expected_columns = get_feature_output_columns()
    existing_columns = list(df_existing.columns)

    if existing_columns != expected_columns:
        print("Error: existing feature CSV schema does not match current expected schema.")
        print(f"Existing columns: {existing_columns}")
        print(f"Expected columns: {expected_columns}")
        print("Please start a new file before resuming feature extraction.")
        raise SystemExit(1)


def load_processed_urls(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()

    df_existing = pd.read_csv(output_path, usecols=["url"])
    return set(df_existing["url"].dropna().astype(str))


def append_rows_to_csv(rows: list[dict], output_path: str):
    if not rows:
        return

    df_out = pd.DataFrame(rows)
    df_out = df_out[get_feature_output_columns()]
    write_header = not os.path.exists(output_path)
    df_out.to_csv(output_path, mode="a", header=write_header, index=False)

    print(
        f"appended {len(rows)} rows to csv at {output_path} "
        f"(buffer flushed, buffer size now 0)"
    )


def build_feature_dataset(df: pd.DataFrame, label_column: str, output_path: str, batch_size_value: int) -> pd.DataFrame:
    validate_existing_feature_csv(output_path)

    processed_urls = load_processed_urls(output_path)
    completed_rows = len(processed_urls)
    remaining_df = df[~df["url"].astype(str).isin(processed_urls)].copy()

    print(f"rows already completed: {completed_rows}")
    print(f"rows remaining: {len(remaining_df)}")

    rows = []
    pbar = tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="Building feature rows")

    for _, row in pbar:
        state = ml_inference({"url": row.url})

        vt_error_text = str(state.get("virustotal", {}).get("error", "") or "")
        if "429" in vt_error_text and "Quota Exceeded" in vt_error_text:
            print(f'VirusTotal quota warning for URL: {row.url} -> "{vt_error_text}"')

        features = build_signal_features(state)
        features["url"] = row.url
        features["label"] = int(getattr(row, label_column))
        rows.append(features)

        pbar.set_postfix(buffer_size=len(rows), completed=completed_rows + len(rows))

        if len(rows) >= batch_size_value:
            append_rows_to_csv(rows, output_path)
            completed_rows += len(rows)
            rows = []
            print(f"rows completed so far: {completed_rows}")
            print(f"rows remaining: {len(df) - completed_rows}")

    if rows:
        append_rows_to_csv(rows, output_path)
        completed_rows += len(rows)
        print(f"rows completed so far: {completed_rows}")
        print(f"rows remaining: {len(df) - completed_rows}")

    return pd.read_csv(output_path)


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


def append_train_results(
    cv_metrics: dict,
    saved_at: str,
    train_time_seconds: float,
    num_samples: int,
    bert_architecture: str,
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

    model_name = f"Stacker model ({os.path.basename(meta_model_dir)})"
    note = f"sklearn v{sklearn.__version__}; {bert_architecture}"

    row = {
        "Model": model_name,
        "Accuracy": cv_metrics.get("Accuracy"),
        "CV Accuracy": "",
        "CV Std": "",
        "Precision": cv_metrics.get("Precision"),
        "Recall": cv_metrics.get("Recall"),
        "F1-Score": cv_metrics.get("F1"),
        "Train Time (s)": round(train_time_seconds, 3),
        "Inference Time (s)": "",
        "Saved_at": saved_at.replace(":", "-"),
        "Training dataset name": training_data_path,
        "ROC-AUC": cv_metrics.get("ROC_AUC"),
        "Num samples in dataset": num_samples,
        "Note": note,
    }

    df_out = pd.DataFrame([[row.get(col) for col in column_order]], columns=column_order)

    try:
        df_existing = pd.read_csv(train_results_output)
        if list(df_existing.columns) != column_order:
            print("Error: all_model_train_results.csv schema does not match expected columns.")
            print(f"Existing columns: {list(df_existing.columns)}")
            print(f"Expected columns: {column_order}")
            print("Skipping append to all_model_train_results.csv.")
            return
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass

    df_out.to_csv(train_results_output, index=False)
    print(f"appended stacker training summary to {train_results_output}")


def main():
    train_start = time.time()
    print("loading training split")
    print("building features")
    df_features = build_feature_dataset(
        df_dev,
        label_column=training_label_column,
        output_path=results_output,
        batch_size_value=batch_size,
    )

    print("running CV")
    cv_metrics = evaluate_stacker_cv(df_features)

    print("fitting final calibrated model")
    model, feature_columns = train_stacker_model(df_features)

    bert_metadata = get_active_bert_metadata()
    saved_at = datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "saved_at": saved_at,
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
        "batch_size": batch_size,
        "feature_csv_path": results_output,
    }

    print("saving artifacts")
    save_stacker_artifacts(model, feature_columns, metadata)
    append_train_results(
        cv_metrics=cv_metrics,
        saved_at=saved_at,
        train_time_seconds=time.time() - train_start,
        num_samples=len(df_features),
        bert_architecture=bert_metadata["bert_architecture"],
    )

    print("Stacker training complete.")
    print(cv_metrics)


if __name__ == "__main__":
    main()
