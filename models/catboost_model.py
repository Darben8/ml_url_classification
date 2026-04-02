from catboost import CatBoostClassifier

_MODEL = None
#model_path = "data/ml_models/catboost_classifier.cbm"
model2_path = "data/ml_models/catboost_crossval_best/catboost_best_fold.cbm"


def load_catboost_model():
    global _MODEL
    if _MODEL is None:
        model = CatBoostClassifier()
        #model.load_model(model_path)
        model.load_model(model2_path)
        _MODEL = model
    return _MODEL


# model = CatBoostClassifier()
# model.load_model("data/ml_models/catboost_classifier.cbm")
