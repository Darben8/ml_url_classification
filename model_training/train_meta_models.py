import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None


feature_set_label = "rich_signal"  # can use rich_signal or 4signal
results_dir = "data/results"
train_results_output = "data/results/all_model_train_results.csv"
ml_models_dir = "data/ml_models"
timezone = "US/Eastern"
random_state = 42
test_size = 0.3
cv_folds = 5


feature_set_config = {
    "rich_signal": {
        "feature_csv_path": "data/results/stacker_training_features.csv",
        "feature_columns": [
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
        ],
        "feature_tag": "rich",
        "display_label": "rich-signal",
    },
    "4signal": {
        "feature_csv_path": "data/results/stacker_training_features_4signal.csv",
        "feature_columns": [
            "bert_score",
            "cb_score",
            "vt_score",
            "tranco_score",
        ],
        "feature_tag": "4signal",
        "display_label": "4-signal",
    },
}


model_specs = {
    "Logistic Regression": {
        "short_name": "lr",
        "builder": lambda: LogisticRegression(max_iter=1000, random_state=random_state),
    },
    "Naive Bayes": {
        "short_name": "nb",
        "builder": lambda: GaussianNB(),
    },
    "Decision Tree": {
        "short_name": "dt",
        "builder": lambda: DecisionTreeClassifier(random_state=random_state, max_depth=10),
    },
    "Random Forest": {
        "short_name": "rf",
        "builder": lambda: RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
    },
    "Gradient Boosting": {
        "short_name": "gb",
        "builder": lambda: GradientBoostingClassifier(n_estimators=100, random_state=random_state),
    },
    "K Nearest Neighbors": {
        "short_name": "knn",
        "builder": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    },
    "CatBoost Classifier": {
        "short_name": "cb",
        "builder": lambda: CatBoostClassifier(verbose=0, random_state=random_state),
    },
    "Support Vector Machine": {
        "short_name": "svm",
        "builder": lambda: SVC(probability=True, random_state=random_state),
    },
    "Multi-layer Perceptron": {
        "short_name": "mlp",
        "builder": lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=random_state),
    },
}

if XGBClassifier is not None:
    model_specs["XGBoost"] = {
        "short_name": "xgb",
        "builder": lambda: XGBClassifier(
            n_estimators=100,
            max_depth=3,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

if LGBMClassifier is not None:
    model_specs["LightGBM"] = {
        "short_name": "lgbm",
        "builder": lambda: LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1),
    }


def get_active_feature_config() -> dict:
    if feature_set_label not in feature_set_config:
        raise ValueError(f"Unsupported feature_set_label: {feature_set_label}")
    return feature_set_config[feature_set_label]


def load_feature_dataset() -> pd.DataFrame:
    config = get_active_feature_config()
    feature_csv_path = config["feature_csv_path"]

    if not os.path.exists(feature_csv_path):
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv_path}")

    df = pd.read_csv(feature_csv_path)
    required_columns = config["feature_columns"] + ["label"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {feature_csv_path}: {missing_columns}")

    return df


def get_model_score_values(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def save_model_artifacts(model_name: str, model, feature_columns: list[str], metrics: dict, saved_at: str):
    config = get_active_feature_config()
    short_name = model_specs[model_name]["short_name"]
    folder_name = f"meta_model_{config['feature_tag']}_{short_name}_v1"
    model_dir = Path(ml_models_dir) / folder_name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "meta_model.pkl")
    joblib.dump(feature_columns, model_dir / "signal_feature_columns.pkl")

    metadata = {
        "saved_at": saved_at,
        "model_name": model_name,
        "feature_set_label": feature_set_label,
        "feature_csv_path": config["feature_csv_path"],
        "feature_columns": feature_columns,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
    }

    with open(model_dir / "meta_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return folder_name


def append_train_results(model_name: str, metrics: dict, saved_at: str, folder_name: str):
    config = get_active_feature_config()
    column_order = [
        "Model",
        "Accuracy",
        "CV Accuracy",
        "CV Std",
        "Precision",
        "Recall",
        "F1-Score",
        "Train Time (s)",
        "Inference Time (s)",
        "Saved_at",
        "Training dataset name",
        "ROC-AUC",
        "Num samples in dataset",
        "Note",
    ]

    row = {
        "Model": f"Meta model ({config['display_label']} {model_name} v1)",
        "Accuracy": metrics["Accuracy"],
        "CV Accuracy": metrics["CV Accuracy"],
        "CV Std": metrics["CV Std"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-Score": metrics["F1"],
        "Train Time (s)": metrics["Train Time (s)"],
        "Inference Time (s)": metrics["Inference Time (s)"],
        "Saved_at": saved_at.replace(":", "-"),
        "Training dataset name": config["feature_csv_path"],
        "ROC-AUC": metrics["ROC_AUC"],
        "Num samples in dataset": metrics["Num Samples"],
        "Note": f"{folder_name}; sklearn v{sklearn.__version__}",
    }

    df_out = pd.DataFrame([[row.get(col) for col in column_order]], columns=column_order)

    try:
        df_existing = pd.read_csv(train_results_output)
        if list(df_existing.columns) != column_order:
            print("Error: all_model_train_results.csv schema does not match expected columns.")
            print("Skipping append to all_model_train_results.csv.")
            return
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass

    df_out.to_csv(train_results_output, index=False)


def train_and_evaluate_model(model_name: str, df: pd.DataFrame) -> dict:
    config = get_active_feature_config()
    feature_columns = config["feature_columns"]
    X = df[feature_columns]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = model_specs[model_name]["builder"]()

    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - train_start, 3)

    inference_start = time.time()
    y_pred = model.predict(X_test)
    inference_time = round(time.time() - inference_start, 3)

    score_values = get_model_score_values(model, X_test)
    cv_scores = cross_val_score(
        model_specs[model_name]["builder"](),
        X_train,
        y_train,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "CV Accuracy": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, score_values),
        "Train Time (s)": train_time,
        "Inference Time (s)": inference_time,
        "Num Samples": len(df),
    }

    return {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
    }


def main():
    config = get_active_feature_config()
    print(f"loading feature dataset: {config['feature_csv_path']}")
    df = load_feature_dataset()
    print(f"feature set label: {feature_set_label}")
    print(f"num samples: {len(df)}")

    saved_at = datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")

    for model_name in tqdm(model_specs.keys(), desc="Training meta models", total=len(model_specs)):
        print(f"training {model_name}")
        result = train_and_evaluate_model(model_name, df)
        folder_name = save_model_artifacts(
            model_name=model_name,
            model=result["model"],
            feature_columns=result["feature_columns"],
            metrics=result["metrics"],
            saved_at=saved_at,
        )
        append_train_results(
            model_name=model_name,
            metrics=result["metrics"],
            saved_at=saved_at,
            folder_name=folder_name,
        )
        print(
            f"{model_name} complete "
            f"(Accuracy: {result['metrics']['Accuracy']:.4f}, F1: {result['metrics']['F1']:.4f})"
        )


if __name__ == "__main__":
    main()
