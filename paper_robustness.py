import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from graph.nodes.ensemble2 import ensemble_decision
from graph.nodes.inference import ml_inference
from graph.nodes.load_data import df_test, df_val
from graph.nodes.stacking_inference import stacking_decision
from models.bert_model import get_active_bert_metadata

# -----------------------------
# Paper experiment configuration
# -----------------------------
OUTPUT_CSV = "data/results/paper_robustness.csv"
TIMEZONE = "US/Eastern"

# Choose which split(s) to run for the paper.
# Common choice: Test only.
DATASETS = {
    #"Validation": df_val,
    "Test": df_test,
}

# Choose your final paper model variant.
# Options in current repo: "4signal" or "rich"
STACKER_VARIANT = "rich"
MAX_URLS_PER_SPLIT = 100

# Robustness settings to evaluate
ROBUSTNESS_SETTINGS = [
    "full",
    "no_vt",
    "no_tranco",
    "no_vt_no_tranco",
]


def map_prediction_to_label(prediction: str) -> int:
    return 1 if prediction == "Benign" else 0


def apply_no_vt(state: dict) -> dict:
    state = dict(state)

    state["vt_error"] = 1
    state["vt_score"] = 0.5
    state["virustotal"] = {
        "vt_malicious_count": None,
        "vt_suspicious_count": None,
        "vt_harmless_count": None,
        "vt_undetected_count": None,
        "vt_total_engines": None,
        "vt_detection_rate": None,
        "error": "Paper robustness test: VT disabled",
    }
    return state


def apply_no_tranco(state: dict) -> dict:
    state = dict(state)

    state["tranco_error"] = 1
    state["tranco_score"] = 0.5
    state["tranco"] = {
        "in_tranco": 0,
        "tranco_rank": None,
        "tranco_score": 0.5,
        "error": "Paper robustness test: Tranco disabled",
    }
    return state


def apply_robustness_setting(state: dict, setting: str) -> dict:
    if setting == "full":
        return state
    if setting == "no_vt":
        return apply_no_vt(state)
    if setting == "no_tranco":
        return apply_no_tranco(state)
    if setting == "no_vt_no_tranco":
        state = apply_no_vt(state)
        state = apply_no_tranco(state)
        return state

    raise ValueError(f"Unsupported robustness setting: {setting}")


def calculate_metrics(y_true: list[int], y_pred: list[int], scores: list[float]) -> dict:
    auc = np.nan if len(set(y_true)) < 2 else roc_auc_score(y_true, scores)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
    }


def run_split_robustness(df: pd.DataFrame, split_name: str, setting: str) -> dict:
    y_true = []
    y_pred = []
    scores = []

    start = time.time()

    for _, row in df.iterrows():
        state = ml_inference({"url": row.url})
        state = ensemble_decision(state)
        state = apply_robustness_setting(state, setting)
        state = stacking_decision(state, stacker_variant=STACKER_VARIANT)

        true_label = int(row.label)
        pred_label = map_prediction_to_label(state["stacking_prediction"])
        score = float(state["stacking_score"])

        y_true.append(true_label)
        y_pred.append(pred_label)
        scores.append(score)

    elapsed = time.time() - start
    metrics = calculate_metrics(y_true, y_pred, scores)

    return {
        "Split": split_name,
        "Robustness Setting": setting,
        "Stacker Variant": STACKER_VARIANT,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "ROC_AUC": metrics["ROC_AUC"],
        "Inference Time (s)": round(elapsed, 3),
        "Avg Time / URL (s)": round(elapsed / len(df), 5),
        "Num Samples": len(df),
    }


def save_results(rows: list[dict], output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df_out = pd.DataFrame(rows)
    df_out["saved_at"] = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")
    df_out["bert_architecture"] = get_active_bert_metadata()["bert_architecture"]

    try:
        df_existing = pd.read_csv(output_csv)
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass

    df_out.to_csv(output_csv, index=False)


def main():
    print(f"Running paper robustness evaluation")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Stacker variant: {STACKER_VARIANT}")

    rows = []

    for split_name, split_df in DATASETS.items():
        if MAX_URLS_PER_SPLIT is not None:
            split_df = split_df.head(MAX_URLS_PER_SPLIT).copy()
        print(f"\n=== {split_name} ({len(split_df)} URLs) ===")
        print(split_df["label"].value_counts())

        for setting in ROBUSTNESS_SETTINGS:
            print(f"Evaluating setting: {setting}")
            row = run_split_robustness(split_df, split_name, setting)
            print(row)
            rows.append(row)

    save_results(rows, OUTPUT_CSV)
    print(f"\nSaved robustness results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()