import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import torch

from graph.nodes.ensemble2 import ensemble_decision
from graph.nodes.inference import ml_inference, tranco_service
from graph.nodes.load_data import df_test, df_val
from graph.nodes.stacking_inference import stacking_decision
from graph.nodes.catboost_inference import catboost_inference
from models.bert_model import get_active_bert_metadata, load_bert_model
from models.meta_model import load_meta_feature_columns, load_meta_model
from models.preprocessing import url_to_tensor
from services.virustotal import vt_check_url
from services.vtcache import get_cached_vt, save_vt_cache, vt_db_path
from utils.normalization import extract_registered_domain


# -----------------------------
# Paper experiment configuration
# -----------------------------
OUTPUT_CSV = "data/results/paper_runtime_analysis.csv"
TIMEZONE = "US/Eastern"

DATASETS = {
    "Validation": df_val,
    "Test": df_test,
}

STACKER_VARIANT = "4signal"
chosen_meta_model_dir = Path("data/ml_models/chosen_meta_models")

VT_MODES = ["cached", "uncached"]
ALLOW_UNCACHED_VT = False
MAX_URLS_PER_SPLIT = None


def _apply_split_limit(df: pd.DataFrame) -> pd.DataFrame:
    if MAX_URLS_PER_SPLIT is None:
        return df
    return df.head(MAX_URLS_PER_SPLIT).copy()


def _delete_vt_cache_row(url: str) -> None:
    conn = sqlite3.connect(vt_db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM vt_cache WHERE url = ?", (url,))
    conn.commit()
    conn.close()


def vt_check_url_uncached(url: str) -> dict:
    original_cache = get_cached_vt(url)

    try:
        _delete_vt_cache_row(url)
        uncached_result = vt_check_url(url)
    finally:
        _delete_vt_cache_row(url)
        if original_cache is not None:
            save_vt_cache(url, original_cache)

    return uncached_result


def run_instrumented_pipeline(url: str, vt_mode: str) -> tuple[dict, dict]:
    if vt_mode == "uncached" and not ALLOW_UNCACHED_VT:
        raise RuntimeError(
            "Uncached VT timing requested but ALLOW_UNCACHED_VT is False. "
            "Set ALLOW_UNCACHED_VT = True to permit live quota-consuming calls."
        )

    state = {
        "url": url,
        "bert_error": 0,
        "catboost_error": 0,
        "vt_error": 0,
        "tranco_error": 0,
    }

    timings = {
        "normalization_time_s": 0.0,
        "tranco_time_s": 0.0,
        "vt_time_s": 0.0,
        "bert_time_s": 0.0,
        "catboost_time_s": 0.0,
        "average_fusion_time_s": 0.0,
        "final_meta_model_time_s": 0.0,
        "full_pipeline_time_s": 0.0,
    }

    pipeline_start = time.perf_counter()

    t0 = time.perf_counter()
    domain = extract_registered_domain(url)
    timings["normalization_time_s"] = time.perf_counter() - t0
    state["normalized_domain"] = domain

    t0 = time.perf_counter()
    if domain:
        try:
            tranco_result = tranco_service.lookup(domain)
        except Exception as exc:
            tranco_result = {
                "in_tranco": 0,
                "tranco_rank": None,
                "tranco_score": 0.5,
                "error": str(exc),
            }
            state["tranco_error"] = 1
    else:
        tranco_result = {
            "in_tranco": 0,
            "tranco_rank": None,
            "tranco_score": 0.5,
            "error": "Could not extract registered domain",
        }
        state["tranco_error"] = 1
    timings["tranco_time_s"] = time.perf_counter() - t0
    state["tranco"] = tranco_result
    state["tranco_score"] = round(float(tranco_result["tranco_score"]), 4)

    t0 = time.perf_counter()
    try:
        if vt_mode == "cached":
            vt_result = vt_check_url(url)
        elif vt_mode == "uncached":
            vt_result = vt_check_url_uncached(url)
        else:
            raise ValueError(f"Unsupported VT mode: {vt_mode}")
    except Exception as exc:
        vt_result = {
            "vt_malicious_count": None,
            "vt_suspicious_count": None,
            "vt_harmless_count": None,
            "vt_undetected_count": None,
            "vt_total_engines": None,
            "vt_detection_rate": None,
            "error": str(exc),
        }
        state["vt_error"] = 1
    timings["vt_time_s"] = time.perf_counter() - t0
    state["virustotal"] = vt_result
    if state["vt_error"]:
        state["vt_score"] = 0.5
    else:
        state["vt_score"] = round(1 - float(vt_result.get("vt_detection_rate", 0.0)), 4)

    t0 = time.perf_counter()
    try:
        model = load_bert_model()
        input_ids, attention_mask = url_to_tensor(url)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=1)
        state["bert_score"] = round(float(probs[:, 1].item()), 4)
    except Exception as exc:
        state["bert_score"] = 0.5
        state["bert_error"] = 1
        state["bert"] = {"error": str(exc)}
    timings["bert_time_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        cb_result = catboost_inference(url)
    except Exception as exc:
        cb_result = {
            "cb_phishing_prob": 0.5,
            "cb_benign_prob": 0.5,
            "cb_prediction": "Uncertain",
            "error": str(exc),
        }
        state["catboost_error"] = 1
    timings["catboost_time_s"] = time.perf_counter() - t0
    state["catboost"] = cb_result
    state["cb_score"] = round(float(cb_result.get("cb_benign_prob", 0.5)), 4)

    t0 = time.perf_counter()
    state = ensemble_decision(state)
    timings["average_fusion_time_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    state = stacking_decision(
        state,
        stacker_variant=STACKER_VARIANT,
        model_dir=str(chosen_meta_model_dir),
    )
    timings["final_meta_model_time_s"] = time.perf_counter() - t0

    timings["full_pipeline_time_s"] = time.perf_counter() - pipeline_start
    return state, timings


def build_base_states(df: pd.DataFrame) -> list[dict]:
    states = []
    for _, row in df.iterrows():
        state = ml_inference({"url": row.url})
        state = ensemble_decision(state)
        states.append(state)
    return states


def measure_decision_layer_only(states: list[dict]) -> dict:
    avg_elapsed = 0.0
    meta_elapsed = 0.0

    load_meta_model(stacker_variant=STACKER_VARIANT, model_dir=str(chosen_meta_model_dir))
    load_meta_feature_columns(stacker_variant=STACKER_VARIANT, model_dir=str(chosen_meta_model_dir))

    for state in states:
        avg_state = dict(state)
        t0 = time.perf_counter()
        ensemble_decision(avg_state)
        avg_elapsed += time.perf_counter() - t0

        meta_state = dict(state)
        t0 = time.perf_counter()
        stacking_decision(
            meta_state,
            stacker_variant=STACKER_VARIANT,
            model_dir=str(chosen_meta_model_dir),
        )
        meta_elapsed += time.perf_counter() - t0

    num_states = len(states)
    return {
        "decision_only_average_total_time_s": round(avg_elapsed, 6),
        "decision_only_average_avg_time_per_url_s": round(avg_elapsed / num_states, 6),
        "decision_only_meta_total_time_s": round(meta_elapsed, 6),
        "decision_only_meta_avg_time_per_url_s": round(meta_elapsed / num_states, 6),
    }


def summarize_component_rows(rows: list[dict], split_name: str, vt_mode: str) -> dict:
    df = pd.DataFrame(rows)

    return {
        "Analysis Type": "full_pipeline_components",
        "Split": split_name,
        "VT Mode": vt_mode,
        "Stacker Variant": STACKER_VARIANT,
        "Selected Meta Model Dir": str(chosen_meta_model_dir),
        "Num Samples": len(df),
        "normalization_time_s": round(df["normalization_time_s"].sum(), 6),
        "normalization_avg_time_per_url_s": round(df["normalization_time_s"].mean(), 6),
        "tranco_time_s": round(df["tranco_time_s"].sum(), 6),
        "tranco_avg_time_per_url_s": round(df["tranco_time_s"].mean(), 6),
        "vt_time_s": round(df["vt_time_s"].sum(), 6),
        "vt_avg_time_per_url_s": round(df["vt_time_s"].mean(), 6),
        "bert_time_s": round(df["bert_time_s"].sum(), 6),
        "bert_avg_time_per_url_s": round(df["bert_time_s"].mean(), 6),
        "catboost_time_s": round(df["catboost_time_s"].sum(), 6),
        "catboost_avg_time_per_url_s": round(df["catboost_time_s"].mean(), 6),
        "average_fusion_time_s": round(df["average_fusion_time_s"].sum(), 6),
        "average_fusion_avg_time_per_url_s": round(df["average_fusion_time_s"].mean(), 6),
        "final_meta_model_time_s": round(df["final_meta_model_time_s"].sum(), 6),
        "final_meta_model_avg_time_per_url_s": round(df["final_meta_model_time_s"].mean(), 6),
        "full_pipeline_time_s": round(df["full_pipeline_time_s"].sum(), 6),
        "full_pipeline_avg_time_per_url_s": round(df["full_pipeline_time_s"].mean(), 6),
    }


def run_full_pipeline_component_analysis(df: pd.DataFrame, split_name: str, vt_mode: str) -> tuple[dict, list[dict]]:
    component_rows = []

    for _, row in df.iterrows():
        _, timings = run_instrumented_pipeline(row.url, vt_mode=vt_mode)
        component_rows.append(timings)

    summary = summarize_component_rows(component_rows, split_name, vt_mode)
    return summary, component_rows


def run_decision_only_analysis(df: pd.DataFrame, split_name: str) -> dict:
    states = build_base_states(df)
    summary = measure_decision_layer_only(states)

    return {
        "Analysis Type": "decision_layer_only",
        "Split": split_name,
        "VT Mode": "precomputed_states",
        "Stacker Variant": STACKER_VARIANT,
        "Selected Meta Model Dir": str(chosen_meta_model_dir),
        "Num Samples": len(states),
        **summary,
    }


def save_results(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df_out = pd.DataFrame(rows)
    df_out["saved_at"] = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")
    df_out["bert_architecture"] = get_active_bert_metadata()["bert_architecture"]

    try:
        df_existing = pd.read_csv(OUTPUT_CSV)
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass

    df_out.to_csv(OUTPUT_CSV, index=False)


def main():
    print("Running paper runtime analysis")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Stacker variant: {STACKER_VARIANT}")
    print(f"Selected meta model dir: {chosen_meta_model_dir}")
    print(f"VT modes: {VT_MODES}")

    if "uncached" in VT_MODES and not ALLOW_UNCACHED_VT:
        print(
            "\nWARNING: 'uncached' VT mode was requested but ALLOW_UNCACHED_VT is False.\n"
            "Set ALLOW_UNCACHED_VT = True before running if you want live uncached VT timings.\n"
        )

    rows = []

    for split_name, split_df in DATASETS.items():
        split_df = _apply_split_limit(split_df)

        print(f"\n=== {split_name} ({len(split_df)} URLs) ===")
        print(split_df["label"].value_counts())

        for vt_mode in VT_MODES:
            if vt_mode == "uncached" and not ALLOW_UNCACHED_VT:
                print("Skipping uncached VT timing because ALLOW_UNCACHED_VT is False")
                continue

            print(f"\nRunning full pipeline + component timing with VT mode: {vt_mode}")
            component_summary, _ = run_full_pipeline_component_analysis(split_df, split_name, vt_mode)
            print(component_summary)
            rows.append(component_summary)

        print("\nRunning decision-layer-only timing on precomputed states")
        decision_summary = run_decision_only_analysis(split_df, split_name)
        print(decision_summary)
        rows.append(decision_summary)

    save_results(rows)
    print(f"\nSaved paper runtime analysis to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
