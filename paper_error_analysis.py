import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from graph.nodes.ensemble2 import ensemble_decision
from graph.nodes.inference import ml_inference
from graph.nodes.load_data import df_test, df_val
from graph.nodes.stacking_inference import stacking_decision
from models.bert_model import get_active_bert_metadata

# -----------------------------
# Paper experiment configuration
# -----------------------------
OUTPUT_ALL_ERRORS = "data/results/paper_error_analysis_all.csv"
OUTPUT_FP = "data/results/paper_error_analysis_false_positives.csv"
OUTPUT_FN = "data/results/paper_error_analysis_false_negatives.csv"

TIMEZONE = "US/Eastern"

# Common choice: Test only
DATASETS = {
    "Validation": df_val,
    "Test": df_test,
}

# Final paper model configuration
STACKER_VARIANT = "4signal"
chosen_meta_model_dir = Path("data/ml_models/chosen_meta_models")

SORT_BY_CONFIDENCE = True
MAX_URLS_PER_SPLIT = None


def map_prediction_to_label(prediction: str) -> int:
    return 1 if prediction == "Benign" else 0


def get_error_type(true_label: int, pred_label: int) -> str:
    # Internal convention:
    # Benign = 1
    # Phishing = 0
    if true_label == 0 and pred_label == 1:
        return "False Positive"  # phishing predicted benign
    if true_label == 1 and pred_label == 0:
        return "False Negative"  # benign predicted phishing
    return "Correct"


def build_error_row(row, state: dict, split_name: str) -> dict:
    true_label = int(row.label)
    pred_label = map_prediction_to_label(state["stacking_prediction"])
    error_type = get_error_type(true_label, pred_label)

    vt_result = state.get("virustotal", {}) or {}
    tranco_result = state.get("tranco", {}) or {}
    catboost_result = state.get("catboost", {}) or {}

    return {
        "Split": split_name,
        "url": row.url,
        "source": getattr(row, "source", None),
        "true_label": true_label,
        "pred_label": pred_label,
        "error_type": error_type,
        "stacking_prediction": state.get("stacking_prediction"),
        "stacking_score": state.get("stacking_score"),
        "ensemble_score": state.get("ensemble_score"),
        "bert_score": state.get("bert_score"),
        "cb_score": state.get("cb_score"),
        "cb_benign_prob": catboost_result.get("cb_benign_prob"),
        "vt_score": state.get("vt_score"),
        "vt_detection_rate": vt_result.get("vt_detection_rate"),
        "vt_malicious_count": vt_result.get("vt_malicious_count"),
        "vt_suspicious_count": vt_result.get("vt_suspicious_count"),
        "vt_total_engines": vt_result.get("vt_total_engines"),
        "tranco_score": state.get("tranco_score"),
        "in_tranco": tranco_result.get("in_tranco"),
        "tranco_rank": tranco_result.get("tranco_rank"),
        "normalized_domain": state.get("normalized_domain"),
        "bert_error": state.get("bert_error"),
        "catboost_error": state.get("catboost_error"),
        "vt_error": state.get("vt_error"),
        "tranco_error": state.get("tranco_error"),
    }


def confidence_distance(score: float) -> float:
    # Larger means farther from the 0.5 threshold, i.e. more confident
    return abs(float(score) - 0.5)


def run_error_analysis(df: pd.DataFrame, split_name: str) -> list[dict]:
    rows = []

    for _, row in df.iterrows():
        state = ml_inference({"url": row.url})
        state = ensemble_decision(state)
        state = stacking_decision(
            state,
            stacker_variant=STACKER_VARIANT,
            model_dir=str(chosen_meta_model_dir),
        )

        error_row = build_error_row(row, state, split_name)
        if error_row["error_type"] != "Correct":
            rows.append(error_row)

    return rows


def save_error_outputs(rows: list[dict]) -> None:
    os.makedirs("data/results", exist_ok=True)

    df_errors = pd.DataFrame(rows)
    df_errors["saved_at"] = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")
    df_errors["bert_architecture"] = get_active_bert_metadata()["bert_architecture"]
    df_errors["stacker_variant"] = STACKER_VARIANT
    df_errors["selected_meta_model_dir"] = str(chosen_meta_model_dir)

    if not df_errors.empty and SORT_BY_CONFIDENCE:
        df_errors["confidence_distance"] = df_errors["stacking_score"].apply(confidence_distance)
        df_errors = df_errors.sort_values(
            by=["error_type", "confidence_distance"],
            ascending=[True, False],
        )

    df_fp = df_errors[df_errors["error_type"] == "False Positive"].copy()
    df_fn = df_errors[df_errors["error_type"] == "False Negative"].copy()

    df_errors.to_csv(OUTPUT_ALL_ERRORS, index=False)
    df_fp.to_csv(OUTPUT_FP, index=False)
    df_fn.to_csv(OUTPUT_FN, index=False)

    print(f"Saved all errors to: {OUTPUT_ALL_ERRORS}")
    print(f"Saved false positives to: {OUTPUT_FP}")
    print(f"Saved false negatives to: {OUTPUT_FN}")
    print(f"Total errors: {len(df_errors)}")
    print(f"False positives: {len(df_fp)}")
    print(f"False negatives: {len(df_fn)}")


def main():
    print("Running paper error analysis")
    print(f"Stacker variant: {STACKER_VARIANT}")
    print(f"Selected meta model dir: {chosen_meta_model_dir}")

    all_rows = []

    for split_name, split_df in DATASETS.items():
        if MAX_URLS_PER_SPLIT is not None:
            split_df = split_df.head(MAX_URLS_PER_SPLIT).copy()
        print(f"\n=== {split_name} ({len(split_df)} URLs) ===")
        print(split_df["label"].value_counts())

        split_rows = run_error_analysis(split_df, split_name)
        print(f"Errors found in {split_name}: {len(split_rows)}")
        all_rows.extend(split_rows)

    save_error_outputs(all_rows)


if __name__ == "__main__":
    main()
