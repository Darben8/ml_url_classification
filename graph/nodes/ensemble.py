## ENSEMBLING SCORES FROM MULTIPLE MODELS AND SERVICES
# This node combines scores from BERT, Tranco, VirusTotal, and CatBoost
# into a final phishing likelihood score using averaging.

from typing import Dict, List
#from unittest import result
threshold = 0.5  # Decision threshold for final classification

# Ensembling when phishing score is 0, benign score is 1
# This is the version for the dataset called new_data_urls.csv where Phishing=0, Benign=1
def mean_ensemble(scores: List[float]) -> float:
    if not scores:
        raise ValueError("No scores provided for ensembling")

    return sum(scores) / len(scores)

def ensemble_decision(result: Dict) -> Dict:
    scores = [
        result["tranco_score"],
        result["vt_score"],
        result["bert_score"],
        result["cb_score"],
    ]

    ensemble_score = mean_ensemble(scores)

    result["ensemble_score"] = round(float(ensemble_score), 4)
    result["std_prediction"] = (
        "Benign" if ensemble_score >= threshold else "Phishing"
    )
    return result


## WEIGHTED ENSEMBLING
def weighted_ensemble_decision(result: Dict) -> Dict:
    weights = {
    "tranco_score": 0.15,
    "vt_score":     0.35,
    "bert_score":   0.25,
    "cb_score":     0.25,
    }

    weighted_score = (
        result["tranco_score"] * weights["tranco_score"] +
        result["vt_score"]     * weights["vt_score"] +
        result["bert_score"]   * weights["bert_score"] +
        result["cb_score"]     * weights["cb_score"]
)

    result["weighted_score"] = round(float(weighted_score), 4)
    # result["final_prediction"] = ("Benign" if weighted_score >= threshold else "Phishing")
    if weighted_score >= 0.7:
        result["weighted_prediction"] = "Benign"
    elif weighted_score <= 0.3:
        result["weighted_prediction"] = "Phishing"
    else:
        result["weighted_prediction"] = "Uncertain"
    return result

