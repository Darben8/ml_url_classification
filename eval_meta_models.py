import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from graph.nodes.ensemble2 import ensemble_decision
from graph.nodes.inference import ml_inference
from graph.nodes.load_data import df_test, df_test_old, df_val, df_val_old
from graph.nodes.stacking_inference import build_4signal_features
from models.bert_model import get_active_bert_metadata
from models.fusion_features import build_signal_features


results_output = "data/results/results.csv"
ml_models_dir = Path("data/ml_models")
timezone = "US/Eastern"
data_type = "new_data"  # can use old_data or new_data
fusion_modes = ["average", "stacking_rich", "stacking_4signal"]


dataset_config = {
    "new_data": {
        "label_column": "label",
        "splits": {
            "Validation": df_val,
            "Test": df_test,
        },
        "display_name": "New Data",
    },
    "old_data": {
        "label_column": "status",
        "splits": {
            "Validation": df_val_old,
            "Test": df_test_old,
        },
        "display_name": "Old Data",
    },
}


@dataclass
class MetaModelSpec:
    model_id: str
    model_name: str
    model_dir: Path
    model_path: Path
    feature_columns: list[str]
    fusion_mode: str
    metadata: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved meta models and append the comparison table to results.csv."
    )
    parser.add_argument(
        "--data-type",
        choices=dataset_config.keys(),
        default=data_type,
        help="Dataset split family to evaluate.",
    )
    parser.add_argument(
        "--fusion-modes",
        nargs="+",
        choices=fusion_modes,
        default=fusion_modes,
        help="Fusion modes to include in the comparison.",
    )
    parser.add_argument(
        "--output",
        default=results_output,
        help="CSV path for the comparison table.",
    )
    return parser.parse_args()


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "meta_model_metadata.json"
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_model_artifact(model_dir: Path) -> Path | None:
    for filename in ["meta_model.pkl", "logistic_regression_calibrated.pkl"]:
        model_path = model_dir / filename
        if model_path.exists():
            return model_path
    return None


def infer_fusion_mode(model_dir: Path, metadata: dict, feature_columns: list[str]) -> str | None:
    feature_set_label = metadata.get("feature_set_label", "")
    model_id = model_dir.name.lower()

    if feature_set_label == "4signal" or "4signal" in model_id:
        return "stacking_4signal"
    if feature_set_label == "rich_signal" or "rich" in model_id:
        return "stacking_rich"
    if len(feature_columns) == 4 and {"bert_score", "cb_score", "vt_score", "tranco_score"}.issubset(feature_columns):
        return "stacking_4signal"
    if "cb_benign_prob" in feature_columns or "vt_detection_rate" in feature_columns:
        return "stacking_rich"
    return None


def infer_model_name(model_path: Path, metadata: dict) -> str:
    if metadata.get("model_name"):
        return metadata["model_name"]
    if model_path.name == "logistic_regression_calibrated.pkl":
        return "Calibrated Logistic Regression"
    return "Unknown Meta Model"


def discover_meta_models(selected_fusion_modes: list[str]) -> list[MetaModelSpec]:
    specs = []
    if not ml_models_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {ml_models_dir}")

    for model_dir in sorted(ml_models_dir.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith("meta_model"):
            continue

        model_path = find_model_artifact(model_dir)
        feature_columns_path = model_dir / "signal_feature_columns.pkl"
        if model_path is None or not feature_columns_path.exists():
            continue

        metadata = load_metadata(model_dir)
        feature_columns = joblib.load(feature_columns_path)
        fusion_mode = infer_fusion_mode(model_dir, metadata, feature_columns)
        if fusion_mode not in selected_fusion_modes:
            continue

        specs.append(
            MetaModelSpec(
                model_id=model_dir.name,
                model_name=infer_model_name(model_path, metadata),
                model_dir=model_dir,
                model_path=model_path,
                feature_columns=feature_columns,
                fusion_mode=fusion_mode,
                metadata=metadata,
            )
        )

    return specs


def build_features_for_fusion(state: dict, fusion_mode: str) -> dict:
    if fusion_mode == "stacking_4signal":
        return build_4signal_features(state)
    if fusion_mode == "stacking_rich":
        return build_signal_features(state)
    raise ValueError(f"Unsupported meta-model fusion mode: {fusion_mode}")


def get_benign_score(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)
        if classes is not None and 1 in classes:
            benign_index = list(classes).index(1)
        else:
            benign_index = 1
        return float(proba[benign_index])

    if hasattr(model, "decision_function"):
        return float(model.decision_function(X)[0])

    return float(model.predict(X)[0])


def compute_base_states(df: pd.DataFrame) -> list[dict]:
    states = []
    for _, row in df.iterrows():
        state = ml_inference({"url": row.url})
        states.append(ensemble_decision(state))
    return states


def map_prediction_to_label(prediction: str) -> int:
    return 1 if prediction == "Benign" else 0


def calculate_metrics(
    y_true: list[int],
    y_pred: list[int],
    scores: list[float],
    elapsed: float,
    split_name: str,
) -> dict:
    auc = np.nan if len(set(y_true)) < 2 else roc_auc_score(y_true, scores)

    return {
        "Split": split_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": round(auc, 4),
        "Inference Time (s)": round(elapsed, 3),
        "Avg Time / URL (s)": round(elapsed / len(y_true), 5),
        "Num Samples": len(y_true),
    }


def evaluate_average(states: list[dict], labels: list[int], split_name: str) -> dict:
    start = time.time()
    scores = [state["ensemble_score"] for state in states]
    y_pred = [map_prediction_to_label(state["std_prediction"]) for state in states]
    elapsed = time.time() - start

    return calculate_metrics(labels, y_pred, scores, elapsed, split_name)


def evaluate_meta_model(
    spec: MetaModelSpec,
    model,
    states: list[dict],
    labels: list[int],
    split_name: str,
) -> dict:
    y_pred = []
    scores = []

    start = time.time()
    for state in states:
        features = build_features_for_fusion(state, spec.fusion_mode)
        X = pd.DataFrame(
            [[features.get(col) for col in spec.feature_columns]],
            columns=spec.feature_columns,
        )
        benign_score = get_benign_score(model, X)
        scores.append(benign_score)
        y_pred.append(1 if benign_score >= 0.5 else 0)

    elapsed = time.time() - start
    return calculate_metrics(labels, y_pred, scores, elapsed, split_name)


def get_training_metrics(metadata: dict) -> dict:
    metrics = metadata.get("metrics") or metadata.get("cv_metrics") or {}

    return {
        "Train Accuracy": metrics.get("Accuracy"),
        "Train CV Accuracy": metrics.get("CV Accuracy"),
        "Train CV Std": metrics.get("CV Std"),
        "Train Precision": metrics.get("Precision"),
        "Train Recall": metrics.get("Recall"),
        "Train F1-Score": metrics.get("F1"),
        "Train Time (s)": metrics.get("Train Time (s)"),
        "Train Inference Time (s)": metrics.get("Inference Time (s)"),
        "Train Saved_at": metadata.get("saved_at"),
        "Training dataset name": metadata.get("feature_csv_path")
        or metadata.get("derived_feature_csv_path")
        or metadata.get("training_data_path"),
        "Train ROC-AUC": metrics.get("ROC_AUC"),
        "Train Num samples in dataset": metrics.get("Num Samples"),
        "Train Note": metadata.get("note"),
    }


def build_result_row(
    metrics: dict,
    fusion_mode: str,
    model_name: str,
    model_id: str,
    model_path: str,
    metadata: dict,
    active_data_type: str,
    bert_architecture: str,
) -> dict:
    row = {
        "ensemble_type": "standard" if fusion_mode == "average" else "n/a",
        "data_type": active_data_type,
        "bert_architecture": bert_architecture,
        "fusion_mode": fusion_mode,
        "Meta Model Name": model_name,
        "Meta Model ID": model_id,
        "Model Artifact": model_path,
        "saved_at": datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    row.update(metrics)
    row.update(get_training_metrics(metadata))
    return row


def save_results(rows: list[dict], output_path: str):
    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        df_existing = pd.read_csv(output_path)
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass

    df_out.to_csv(output_path, index=False)


def print_configuration(
    active_data_type: str,
    selected_fusion_modes: list[str],
    model_specs: list[MetaModelSpec],
    output_path: str,
):
    print(f"Data type: {active_data_type}")
    print(f"Fusion modes: {', '.join(selected_fusion_modes)}")
    print(f"Discovered compatible meta models: {len(model_specs)}")
    print(f"Results output: {output_path}")


def main():
    args = parse_args()
    active_dataset = dataset_config[args.data_type]
    selected_fusion_modes = args.fusion_modes
    bert_architecture = get_active_bert_metadata()["bert_architecture"]
    model_specs = discover_meta_models(selected_fusion_modes)

    print_configuration(args.data_type, selected_fusion_modes, model_specs, args.output)

    loaded_models = {
        spec.model_id: joblib.load(spec.model_path)
        for spec in model_specs
    }

    rows = []
    for split_name, split_df in active_dataset["splits"].items():
        print(f"\nBuilding base inference states for {split_name} ({len(split_df)} URLs)")
        states = compute_base_states(split_df)
        labels = split_df[active_dataset["label_column"]].astype(int).tolist()

        if "average" in selected_fusion_modes:
            print(f"Evaluating average ensemble on {split_name}")
            metrics = evaluate_average(states, labels, split_name)
            rows.append(
                build_result_row(
                    metrics=metrics,
                    fusion_mode="average",
                    model_name="Standard Average Ensemble",
                    model_id="n/a",
                    model_path="n/a",
                    metadata={},
                    active_data_type=args.data_type,
                    bert_architecture=bert_architecture,
                )
            )

        for spec in model_specs:
            print(f"Evaluating {spec.model_id} on {split_name}")
            metrics = evaluate_meta_model(
                spec=spec,
                model=loaded_models[spec.model_id],
                states=states,
                labels=labels,
                split_name=split_name,
            )
            rows.append(
                build_result_row(
                    metrics=metrics,
                    fusion_mode=spec.fusion_mode,
                    model_name=spec.model_name,
                    model_id=spec.model_id,
                    model_path=str(spec.model_path),
                    metadata=spec.metadata,
                    active_data_type=args.data_type,
                    bert_architecture=bert_architecture,
                )
            )

    save_results(rows, args.output)
    print(f"\nSaved {len(rows)} comparison rows to {args.output}")


if __name__ == "__main__":
    main()
