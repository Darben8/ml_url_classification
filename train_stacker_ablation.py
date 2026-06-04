import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


TIMEZONE = "US/Eastern"
DEFAULT_OUTPUT_DIR = Path("data/ml_models/ablation")
TRAIN_RESULTS_OUTPUT = "data/results/all_model_train_results.csv"

ABLATION_GROUP_CONFIG = {
    "4sig": {
        "feature_csv_path": "data/results/stacker_training_features_4signal.csv",
        "ablation_mode": "strict",
        "feature_set_label": "4sig",
        "parent_feature_dataset": "data/results/stacker_training_features_4signal.csv",
        "feature_set_description": "Compact four-signal feature dataset derived from ensemble signals.",
    },
    "rich": {
        "feature_csv_path": "data/results/stacker_training_features.csv",
        "ablation_mode": "strict",
        "feature_set_label": "rich",
        "parent_feature_dataset": "data/results/stacker_training_features.csv",
        "feature_set_description": "Strict rich-signal ablations with entire signal families and error flags removed.",
    },
    "richops": {
        "feature_csv_path": "data/results/stacker_training_features.csv",
        "ablation_mode": "operational",
        "feature_set_label": "richops",
        "parent_feature_dataset": "data/results/stacker_training_features.csv",
        "feature_set_description": "Operational rich-signal ablations that preserve relevant error flags.",
    },
}

ABLATION_COLUMN_CONFIG = {
    "4sig": {
        "intrinsic_only": ["bert_score", "cb_score"],
        "intrinsic_and_vt": ["bert_score", "cb_score", "vt_score"],
        "intrinsic_and_tranco": ["bert_score", "cb_score", "tranco_score"],
        "extrinsic": ["vt_score", "tranco_score"],
    },
    "rich": {
        "intrinsic_only": ["bert_score", "cb_benign_prob"],
        "intrinsic_and_vt": [
            "bert_score",
            "cb_benign_prob",
            "vt_detection_rate",
            "vt_malicious_count",
            "vt_suspicious_count",
            "vt_total_engines",
        ],
        "intrinsic_and_tranco": [
            "bert_score",
            "cb_benign_prob",
            "in_tranco",
            "tranco_score",
            "tranco_rank",
        ],
        "extrinsic": [
            "vt_detection_rate",
            "vt_malicious_count",
            "vt_suspicious_count",
            "vt_total_engines",
            "in_tranco",
            "tranco_score",
            "tranco_rank",
        ],
    },
    "richops": {
        "intrinsic_only": [
            "bert_score",
            "cb_benign_prob",
            "bert_error",
            "catboost_error",
        ],
        "intrinsic_and_vt": [
            "bert_score",
            "cb_benign_prob",
            "vt_detection_rate",
            "vt_malicious_count",
            "vt_suspicious_count",
            "vt_total_engines",
            "bert_error",
            "catboost_error",
            "vt_error",
        ],
        "intrinsic_and_tranco": [
            "bert_score",
            "cb_benign_prob",
            "in_tranco",
            "tranco_score",
            "tranco_rank",
            "bert_error",
            "catboost_error",
            "tranco_error",
        ],
        "extrinsic": [
            "vt_detection_rate",
            "vt_malicious_count",
            "vt_suspicious_count",
            "vt_total_engines",
            "in_tranco",
            "tranco_score",
            "tranco_rank",
            "vt_error",
            "tranco_error",
        ],
    },
}

MODEL_DISPLAY_NAMES = {
    "gb": "Gradient Boosting",
    "dt": "Decision Tree",
    "lr": "Logistic Regression",
    "svm": "Support Vector Machine",
    "mlp": "Multi-layer Perceptron",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ablation meta models from selected feature subsets."
    )
    parser.add_argument(
        "--ablation-group",
        nargs="+",
        choices=sorted(ABLATION_GROUP_CONFIG.keys()),
        default=sorted(ABLATION_GROUP_CONFIG.keys()),
        help="Feature-set family or families to train.",
    )
    parser.add_argument(
        "--ablation-name",
        nargs="+",
        choices=sorted(next(iter(ABLATION_COLUMN_CONFIG.values())).keys()),
        default=sorted(next(iter(ABLATION_COLUMN_CONFIG.values())).keys()),
        help="Ablation subset(s) to train.",
    )
    parser.add_argument(
        "--model-name",
        nargs="+",
        choices=sorted(MODEL_DISPLAY_NAMES.keys()),
        default=sorted(MODEL_DISPLAY_NAMES.keys()),
        help="Meta-model family or families to train.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved ablation model artifacts.",
    )
    parser.add_argument(
        "--train-results-output",
        default=TRAIN_RESULTS_OUTPUT,
        help="CSV path for training summaries.",
    )
    return parser.parse_args()


def get_parent_feature_config(ablation_group: str) -> dict:
    if ablation_group not in ABLATION_GROUP_CONFIG:
        raise ValueError(f"Unsupported ablation_group: {ablation_group}")
    return ABLATION_GROUP_CONFIG[ablation_group]


def get_ablation_feature_columns(ablation_group: str, ablation_name: str) -> list[str]:
    try:
        return ABLATION_COLUMN_CONFIG[ablation_group][ablation_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported ablation combination: {ablation_group}/{ablation_name}"
        ) from exc


def get_feature_set_description(ablation_group: str, ablation_name: str, feature_columns: list[str]) -> str:
    parent_description = ABLATION_GROUP_CONFIG[ablation_group]["feature_set_description"]
    joined = ", ".join(feature_columns)
    return f"{parent_description} Ablation '{ablation_name}' uses: {joined}."


def build_training_pipeline(model_name: str):
    if model_name == "lr":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    if model_name == "svm":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, class_weight="balanced")),
            ]
        )

    if model_name == "mlp":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
            ]
        )

    if model_name == "dt":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42)),
            ]
        )

    if model_name == "gb":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("clf", GradientBoostingClassifier(random_state=42)),
            ]
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def validate_feature_schema(df: pd.DataFrame, feature_columns: list[str], source_path: str):
    required_columns = feature_columns + ["url", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in {source_path}: {missing_columns}")


def evaluate_cv(model, X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    probas = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    benign_scores = probas[:, 1]
    y_pred = (benign_scores >= 0.5).astype(int)

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "CV Accuracy": float(accuracy_scores.mean()),
        "CV Std": float(accuracy_scores.std()),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, benign_scores),
        "Num Samples": int(len(X)),
    }


def measure_inference_time(model, X: pd.DataFrame) -> float:
    start = time.time()
    model.predict_proba(X)
    return round(time.time() - start, 3)


def save_artifacts(output_dir: Path, feature_columns: list[str], model, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "meta_model.pkl")
    joblib.dump(feature_columns, output_dir / "signal_feature_columns.pkl")
    with (output_dir / "meta_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def append_train_results(row: dict, output_path: str):
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
        "Ablation Group",
        "Ablation Name",
        "Ablation Mode",
        "Parent Feature Dataset",
    ]

    df_out = pd.DataFrame([[row.get(col) for col in column_order]], columns=column_order)

    try:
        df_existing = pd.read_csv(output_path)
        df_existing = df_existing.reindex(columns=column_order)
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)


def train_single_combination(
    ablation_group: str,
    ablation_name: str,
    model_name: str,
    output_root: Path,
    train_results_output: str,
):
    parent_config = get_parent_feature_config(ablation_group)
    feature_csv_path = parent_config["feature_csv_path"]
    feature_columns = get_ablation_feature_columns(ablation_group, ablation_name)
    feature_set_description = get_feature_set_description(ablation_group, ablation_name, feature_columns)

    df = pd.read_csv(feature_csv_path)
    validate_feature_schema(df, feature_columns, feature_csv_path)

    X = df[feature_columns]
    y = df["label"].astype(int)

    model = build_training_pipeline(model_name)
    display_name = MODEL_DISPLAY_NAMES[model_name]
    model_id = f"meta_model_{ablation_group}_{ablation_name}_{model_name}"
    model_dir = output_root / model_id

    train_start = time.time()
    metrics = evaluate_cv(model, X, y)
    model.fit(X, y)
    train_time = round(time.time() - train_start, 3)
    inference_time = measure_inference_time(model, X)

    saved_at = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")
    metrics["Train Time (s)"] = train_time
    metrics["Inference Time (s)"] = inference_time

    metadata = {
        "saved_at": saved_at,
        "model_name": display_name,
        "model_id": model_id,
        "feature_set_label": ablation_group,
        "feature_set_description": feature_set_description,
        "feature_columns": feature_columns,
        "feature_csv_path": feature_csv_path,
        "ablation_group": ablation_group,
        "ablation_name": ablation_name,
        "ablation_mode": parent_config["ablation_mode"],
        "parent_feature_dataset": parent_config["parent_feature_dataset"],
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
    }

    save_artifacts(model_dir, feature_columns, model, metadata)

    append_train_results(
        {
            "Model": f"Ablation meta model ({display_name})",
            "Accuracy": metrics["Accuracy"],
            "CV Accuracy": metrics["CV Accuracy"],
            "CV Std": metrics["CV Std"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1"],
            "Train Time (s)": train_time,
            "Inference Time (s)": inference_time,
            "Saved_at": saved_at.replace(":", "-"),
            "Training dataset name": feature_csv_path,
            "ROC-AUC": metrics["ROC_AUC"],
            "Num samples in dataset": metrics["Num Samples"],
            "Note": f"{model_id}; sklearn v{sklearn.__version__}",
            "Ablation Group": ablation_group,
            "Ablation Name": ablation_name,
            "Ablation Mode": parent_config["ablation_mode"],
            "Parent Feature Dataset": parent_config["parent_feature_dataset"],
        },
        train_results_output,
    )

    print(
        f"trained {model_id} on {feature_csv_path} "
        f"with columns: {', '.join(feature_columns)}"
    )


def main():
    args = parse_args()
    output_root = Path(args.output_dir)

    print("Training ablation meta models")
    print(f"Output directory: {output_root}")
    print(f"Training summary CSV: {args.train_results_output}")

    for ablation_group in args.ablation_group:
        for ablation_name in args.ablation_name:
            for model_name in args.model_name:
                train_single_combination(
                    ablation_group=ablation_group,
                    ablation_name=ablation_name,
                    model_name=model_name,
                    output_root=output_root,
                    train_results_output=args.train_results_output,
                )


if __name__ == "__main__":
    main()
