import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve

from graph.nodes.ensemble2 import ensemble_decision, weighted_ensemble_decision
from graph.nodes.inference import ml_inference
from graph.nodes.load_data import (
    df_dev,
    df_dev_old,
    df_test,
    df_test_old,
    df_val,
    df_val_old,
)
from models.bert_model import get_active_bert_metadata


results_output = "data/results/results.csv"
fig_dir = "data/figures"
timezone = "US/Eastern"
data_type = "new_data" #can use old_data or new_data
use_weighted = False

os.makedirs(fig_dir, exist_ok=True)

bert_metadata = get_active_bert_metadata()
bert_architecture = bert_metadata["bert_architecture"]
ensemble_type = "weighted" if use_weighted else "standard"

dataset_config = {
    "new_data": {
        "label_column": "label",
        "splits": {
            "Validation": df_val,
            "Test": df_test,
        },
        "display_name": "New Data",
        "file_tag": "new_data",
    },
    "old_data": {
        "label_column": "status",
        "splits": {
            "Validation": df_val_old,
            "Test": df_test_old,
        },
        "display_name": "Old Data",
        "file_tag": "old_data",
    },
}


if data_type not in dataset_config:
    raise ValueError(f"Unsupported data_type: {data_type}")


active_dataset = dataset_config[data_type]


def map_prediction_to_label(status: str) -> int:
    return 1 if status == "Benign" else 0


def get_prediction_fields() -> tuple[str, str]:
    if use_weighted:
        return "weighted_prediction", "weighted_score"
    return "std_prediction", "ensemble_score"


def run_split_evaluation(df: pd.DataFrame, split_name: str):
    y_true, y_pred, scores = [], [], []
    start = time.time()

    prediction_field, score_field = get_prediction_fields()
    label_column = active_dataset["label_column"]

    for _, row in df.iterrows():
        result = ml_inference({"url": row.url})
        result = ensemble_decision(result)

        if use_weighted:
            result = weighted_ensemble_decision(result)

        pred_label = result[prediction_field]
        score = result[score_field]

        y_true.append(int(getattr(row, label_column)))
        y_pred.append(map_prediction_to_label(pred_label))
        scores.append(score)

    elapsed = time.time() - start
    auc = np.nan if len(set(y_true)) < 2 else roc_auc_score(y_true, scores)

    metrics = {
        "Split": split_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": round(auc, 4),
        "Inference Time (s)": round(elapsed, 2),
        "Avg Time / URL (s)": round(elapsed / len(df), 4),
        "Num Samples": len(df),
    }

    return metrics, np.array(scores), np.array(y_true)


def build_figure_filename(split_name: str) -> str:
    ts = datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d_%H-%M-%S")
    safe_split = split_name.lower().replace(" ", "_")
    return (
        f"{fig_dir}/{safe_split}_{active_dataset['file_tag']}_"
        f"{ensemble_type}_{bert_architecture}_{ts}.png"
    )


def plot_score_distribution(scores, labels, split_name: str):
    filename = build_figure_filename(f"{split_name}_scores")
    title = (
        f"{split_name} Score Distribution "
        f"({active_dataset['display_name']}, {ensemble_type}, {bert_architecture})"
    )

    plt.figure(figsize=(7, 4))
    plt.hist(scores[labels == 1], bins=30, alpha=0.6, label="Benign", density=True)
    plt.hist(scores[labels == 0], bins=30, alpha=0.6, label="Phishing", density=True)
    plt.xlabel("Weighted Ensemble Score" if use_weighted else "Standard Ensemble Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_roc_curve(y_true, scores, split_name: str):
    filename = build_figure_filename(f"{split_name}_roc")
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"ROC Curve - {split_name} "
        f"({active_dataset['display_name']}, {ensemble_type}, {bert_architecture})"
    )
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


def save_metrics(metrics: dict):
    timestamp = datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")
    metrics["saved_at"] = timestamp
    metrics["ensemble_type"] = ensemble_type
    metrics["data_type"] = data_type
    metrics["bert_architecture"] = bert_architecture

    column_order = [
        "Split",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "Inference Time (s)",
        "Avg Time / URL (s)",
        "Num Samples",
        "saved_at",
        "ensemble_type",
        "data_type",
        "bert_architecture",
    ]

    df_out = pd.DataFrame([[metrics.get(col) for col in column_order]], columns=column_order)

    try:
        df_existing = pd.read_csv(results_output)
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass

    df_out.to_csv(results_output, index=False)


def print_active_configuration():
    print(f"Data type: {data_type}")
    print(f"Ensemble type: {ensemble_type}")
    print(f"BERT architecture: {bert_architecture}")
    print(f"Results output: {results_output}")

    for split_name, split_df in active_dataset["splits"].items():
        label_column = active_dataset["label_column"]
        print(f"\n{split_name} label counts:")
        print(split_df[label_column].value_counts())


if __name__ == "__main__":
    print_active_configuration()

    for split_name, split_df in active_dataset["splits"].items():
        print(
            f"\n--- {split_name.upper()} SET EVALUATION "
            f"({active_dataset['display_name']}, {ensemble_type}) ---"
        )

        split_metrics, split_scores, split_labels = run_split_evaluation(split_df, split_name)
        print(split_metrics)
        save_metrics(split_metrics)
        plot_score_distribution(split_scores, split_labels, split_name.lower())
        plot_roc_curve(split_labels, split_scores, split_name.lower())
