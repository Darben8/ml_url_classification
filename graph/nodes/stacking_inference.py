from models.fusion_features import build_signal_features
from models.meta_model import predict_meta_model


def build_4signal_features(state: dict) -> dict:
    vt_result = state.get("virustotal", {}) or {}
    vt_detection_rate = vt_result.get("vt_detection_rate")

    vt_score = float("nan")
    if vt_detection_rate is not None:
        vt_score = 1 - float(vt_detection_rate)

    return {
        "bert_score": float(state.get("bert_score", 0.5) or 0.5),
        "cb_score": float(state.get("cb_score", 0.5) or 0.5),
        "vt_score": vt_score,
        "tranco_score": float(state.get("tranco_score", 0.5) or 0.5),
    }


def stacking_decision(state: dict, stacker_variant: str = "rich") -> dict:
    if stacker_variant == "4signal":
        features = build_4signal_features(state)
    else:
        features = build_signal_features(state)

    stacking_result = predict_meta_model(features, stacker_variant=stacker_variant)

    state["stacking_features"] = features
    state.update(stacking_result)
    return state
