import torch
from graph.nodes import catboost_inference
from models.bert_model import load_bert_model
from models.preprocessing import url_to_tensor
from services.tranco import TrancoService
from services.virustotal import vt_check_url
from utils.normalization import extract_registered_domain
from graph.nodes.catboost_inference import catboost_inference
from graph.nodes.ensemble2 import ensemble_decision
from graph.nodes.ensemble2 import weighted_ensemble_decision

# Phishing score: 0, Benign score: 1

tranco_service = TrancoService()


def ml_inference(state: dict):
    """
    LangGraph node for ML inference on a single URL.
    Args:
        state (dict): Must contain a "url" key with the URL string.
    Returns:
        state (dict): State enriched with ML, reputation and lexical url features.
    """
    url = state.get("url")
    if url is None:
        raise ValueError("State dictionary must contain 'url' key")

    state["bert_error"] = 0
    state["catboost_error"] = 0
    state["vt_error"] = 0
    state["tranco_error"] = 0

    domain = extract_registered_domain(url)
    state["normalized_domain"] = domain

    if domain:
        try:
            tranco_result = tranco_service.lookup(domain)
        except Exception as e:
            tranco_result = {
                "in_tranco": 0,
                "tranco_rank": None,
                "tranco_score": 0.5,
                "error": str(e),
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

    state["tranco"] = tranco_result
    state["tranco_score"] = round(float(tranco_result["tranco_score"]), 4)

    try:
        vt_result = vt_check_url(url)
    except Exception as e:
        vt_result = {
            "vt_malicious_count": None,
            "vt_suspicious_count": None,
            "vt_harmless_count": None,
            "vt_undetected_count": None,
            "vt_total_engines": None,
            "vt_detection_rate": None,
            "error": str(e),
        }
        state["vt_error"] = 1

    state["virustotal"] = vt_result
    if state["vt_error"]:
        state["vt_score"] = 0.5
    else:
        state["vt_score"] = round(1 - float(vt_result.get("vt_detection_rate", 0.0)), 4)

    try:
        model = load_bert_model()
        input_ids, attention_mask = url_to_tensor(url)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=1)

        state["bert_score"] = round(float(probs[:, 1].item()), 4)
    except Exception as e:
        state["bert_score"] = 0.5
        state["bert_error"] = 1
        state["bert"] = {"error": str(e)}

    try:
        cb_result = catboost_inference(url)
    except Exception as e:
        cb_result = {
            "cb_phishing_prob": 0.5,
            "cb_benign_prob": 0.5,
            "cb_prediction": "Uncertain",
            "error": str(e),
        }
        state["catboost_error"] = 1

    state["catboost"] = cb_result
    state["cb_score"] = round(float(cb_result.get("cb_benign_prob", 0.5)), 4)

    label = ensemble_decision(state)
    state.update(label)

    weighted_label = weighted_ensemble_decision(state)
    state.update(weighted_label)

    return state
