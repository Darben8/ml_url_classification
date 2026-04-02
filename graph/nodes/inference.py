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

#Phishing score: 0, Benign score: 1

tranco_service = TrancoService()

def ml_inference(state: dict):
    """
    LangGraph node for ML inference on a single URL.
    Args:
        state (dict): Must contain a "url" key with the URL string.
    Returns:
        state (dict): State enriched with ML, reputation and lexical url features.
    """
    # 1. Load model
    model = load_bert_model()

    # 2. Extract URL from state
    url = state.get("url")
    if url is None:
        raise ValueError("State dictionary must contain 'url' key")

    # 3. Normalize URL → registered domain (eTLD+1)
    domain = extract_registered_domain(url)
    state["normalized_domain"] = domain

    # 4. Tranco lookup (non-blocking, non-fatal)
    if domain:
        tranco_result = tranco_service.lookup(domain)
    else:
        tranco_result = {
            "in_tranco": 0,
            "tranco_rank": None,
            "tranco_score": 0.0,
        }
    state["tranco"] = tranco_result
    state["tranco_score"] = round(tranco_result["tranco_score"], 4)

    # 5. VirusTotal lookup
    try:
        vt_result = vt_check_url(url)
    except Exception as e:
        vt_result = {
        "vt_malicious_count": 0,
        "vt_suspicious_count": 0,
        "vt_harmless_count": 0,
        "vt_undetected_count": 0,
        "vt_total_engines": 0,
        "vt_detection_rate": 0.0,
        "error": str(e)
        }
    state["virustotal"] = vt_result
    state["vt_score"] = round(1 - float(vt_result.get("vt_detection_rate", 0.0)), 4)

    # 6. Preprocess URL for BERT → input tensor + attention mask
    input_ids, attention_mask = url_to_tensor(url)

    # 7. Forward pass (BERT inference)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(output.logits, dim=1)

    # 8. Add scores to state
    state["bert_score"] = round(float(probs[:, 1].item()), 4)  # probability of Benign
    #state["bert_phishing_score"] = float(probs[:, 0].item())  # optional: probability of Phishing
    

    # # 9. URL lexical features
    # url_feats = extract_url_features(url)
    # #state.update(url_feats)
    # state["url_features"] = url_feats

    # 10. CatBoost inference
    try:
        cb_result = catboost_inference(url)
    except Exception as e:
        cb_result = {"error": str(e)}

    state["catboost"] = cb_result
    state["cb_score"] = round(float(cb_result.get("cb_benign_prob", 0.0)), 4)

    # 11. Ensemble decision
    label = ensemble_decision(state)
    state.update(label)

    #12. Weighted ensemble decision
    weighted_label = weighted_ensemble_decision(state)
    state.update(weighted_label)

    return state

    
    



