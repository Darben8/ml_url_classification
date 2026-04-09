from models.fusion_features import build_signal_features
from models.meta_model import predict_meta_model


def stacking_decision(state: dict) -> dict:
    features = build_signal_features(state)
    stacking_result = predict_meta_model(features)

    state["stacking_features"] = features
    state.update(stacking_result)
    return state
