import pandas as pd
from models.catboost_model import load_catboost_model
from services.url_features import extract_url_features
import joblib

feature_cols_path = "data/ml_models/feature_columns.pkl"
#feature_cols_path = "data/ml_models/catboost_crossval_best/charbert_optimized_best_fold.pkl" #feature columns from the best catboost fold in cross validation
feature_list = joblib.load(feature_cols_path)

#inference when phishing score is 0, benign score is 1
#This is the version for the dataset called new_data_urls.csv where Phishing=0, Benign=1
def catboost_inference(url: str) -> dict:
    model = load_catboost_model()

    features = extract_url_features(url)
    if features is None:
        raise ValueError("Could not extract URL features")

    # print("Expected features:", sorted(feature_list))
    # print("Extracted features:", sorted(features.keys()))

    X = pd.DataFrame([features])[feature_list]

    # CatBoost probability output: [[p_class0, p_class1]]
    proba = model.predict_proba(X)[0]

    phishing_prob = float(proba[0])
    benign_prob = float(proba[1])

    #prediction = "Phishing" if phishing_prob > benign_prob else "Benign"
    prediction = "Benign" if benign_prob >= 0.5 else "Phishing"

    return {
        "model": "CatBoostClassifier",
        "cb_phishing_prob": phishing_prob,
        "cb_benign_prob": benign_prob,
        "cb_prediction": prediction,
    }
