def get_signal_feature_columns() -> list[str]:
    return [
        "bert_score",
        "cb_benign_prob",
        "vt_detection_rate",
        "vt_malicious_count",
        "vt_suspicious_count",
        "vt_total_engines",
        "in_tranco",
        "tranco_score",
        "tranco_rank",
        "bert_error",
        "catboost_error",
        "vt_error",
        "tranco_error",
    ]


def build_signal_features(state: dict) -> dict:
    vt_result = state.get("virustotal", {}) or {}
    tranco_result = state.get("tranco", {}) or {}
    catboost_result = state.get("catboost", {}) or {}

    tranco_rank = tranco_result.get("tranco_rank")
    if state.get("tranco_error", 0):
        tranco_rank_value = float("nan")
    elif tranco_rank is None:
        tranco_rank_value = 1_000_001
    else:
        tranco_rank_value = int(tranco_rank)

    return {
        "bert_score": float(state.get("bert_score", 0.5) or 0.5),
        "cb_benign_prob": float(catboost_result.get("cb_benign_prob", state.get("cb_score", 0.5)) or 0.5),
        "vt_detection_rate": float(vt_result["vt_detection_rate"]) if vt_result.get("vt_detection_rate") is not None else float("nan"),
        "vt_malicious_count": int(vt_result["vt_malicious_count"]) if vt_result.get("vt_malicious_count") is not None else float("nan"),
        "vt_suspicious_count": int(vt_result["vt_suspicious_count"]) if vt_result.get("vt_suspicious_count") is not None else float("nan"),
        "vt_total_engines": int(vt_result["vt_total_engines"]) if vt_result.get("vt_total_engines") is not None else float("nan"),
        "in_tranco": int(tranco_result.get("in_tranco", 0) or 0),
        "tranco_score": float(tranco_result.get("tranco_score", state.get("tranco_score", 0.5)) or 0.5),
        "tranco_rank": tranco_rank_value,
        "bert_error": int(state.get("bert_error", 0) or 0),
        "catboost_error": int(state.get("catboost_error", 0) or 0),
        "vt_error": int(state.get("vt_error", 0) or 0),
        "tranco_error": int(state.get("tranco_error", 0) or 0),
    }
