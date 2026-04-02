#new evaluation file
from cProfile import label
import os
import time
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from graph.nodes.inference import ml_inference
from graph.nodes.ensemble2 import ensemble_decision, weighted_ensemble_decision
from graph.nodes.load_data import df_dev, df_val, df_test
#from graph.nodes.load_data import df_dev_old, df_val_old, df_test_old

#correct/ most recent test of evaluation of entire ensemble
metrics_output = "data/results/eval_results_new_data_cv3f3.csv" #cvf = cross validation fold, this is the best fold from catboost & bertcross validation
#metrics_output = "data/results/eval_results_old_data_cv3f3.csv" #same cross validation fold but evaluated on old (or training) dataset instead of new dataset
fig_dir = "data/figures"
use_weighted = False #True when you want to use weighted_ensemble score instead of the standard
timezone = "US/Eastern"
os.makedirs(fig_dir, exist_ok=True)

#checking class counts and dataset split
print(df_val["label"].value_counts())
print(df_test["label"].value_counts())

# print(df_val_old["status"].value_counts())
# print(df_test_old["status"].value_counts())

# def map_weighted_label(label: str) -> int:
#     return 1 if label == "Benign" else 0

def map_weighted_label(status: str) -> int:
    # Uncertain → Phishing (0)
    return 1 if status == "Benign" else 0

def run_split_evaluation(df, split_name: str):
    y_true, y_pred = [], []
    scores = []
    start = time.time()

    #print(df.columns)
    for _, row in df.iterrows():
        result = ml_inference({"url": row.url})
        result = ensemble_decision(result)

        if use_weighted:
            result = weighted_ensemble_decision(result)
            pred_label = result["weighted_prediction"]
            score = result["weighted_score"]
        else:
            pred_label = result["std_prediction"]
            score = result["ensemble_score"]


        y_true.append(int(row.label)) #column for new dataset
        #y_true.append(int(row.status)) #column for old dataset
        y_pred.append(map_weighted_label(pred_label))
        scores.append(score)

        #auc = roc_auc_score(y_true, scores)

    elapsed = time.time() - start
    if len(set(y_true)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, scores)


    metrics = {
        "Split": split_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": round(auc,4),
        "Inference Time (s)": round(elapsed, 2),
        "Avg Time / URL (s)": round(elapsed / len(df), 4),
        "Num Samples": len(df)
    }

    return metrics, np.array(scores), np.array(y_true)

def build_figure_filename(split_name: str, ensemble_type: str):
    ts = datetime.now(ZoneInfo("US/Eastern")).strftime("%Y-%m-%d_%H-%M-%S")
    return f"data/figures/{split_name.lower()}_{ensemble_type}_{ts}.png"

def plot_score_distribution(scores, labels, split_name, title):
    ensemble_type = "weighted" if use_weighted else "standard"
    filename = build_figure_filename(split_name, ensemble_type)

    plt.figure(figsize=(7, 4))
    plt.hist(scores[labels == 1], bins=30, alpha=0.6, label="Benign", density=True)
    plt.hist(scores[labels == 0], bins=30, alpha=0.6, label="Phishing", density=True)

    plt.xlabel("Fixed Ensemble Score" if not use_weighted else "Weighted Ensemble Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_roc_curve(y_true, scores, split_name):
    ensemble_type = "weighted" if use_weighted else "standard"

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    filename = build_figure_filename(f"{split_name}_roc", ensemble_type)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {split_name.capitalize()} ({ensemble_type})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

    return auc


def save_metrics(metrics: dict):
    timestamp = datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")
    metrics["saved_at"] = timestamp
    metrics["ensemble_type"] = "weighted" if use_weighted else "standard"

    df_out = pd.DataFrame([metrics])

    # Append safely
    try:
        df_existing = pd.read_csv(metrics_output)
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass

    df_out.to_csv(metrics_output, index=False)


# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("\n--- VALIDATION SET EVALUATION ON OLD TRAINING DATA ---")
    #val_metrics, val_scores, val_labels = run_split_evaluation(df_val_old, "Validation")
    val_metrics, val_scores, val_labels = run_split_evaluation(df_val, "Validation")
    print(val_metrics)
    save_metrics(val_metrics)

    plot_score_distribution(
        val_scores,
        val_labels,
        "validation",
        "Validation Score Distribution",
        #f"{fig_dir}/validation_scores_{int(time.time())}.png",
    )
    plot_roc_curve(
    val_labels,
    val_scores,
    "old data validation",
    )

    print("\n--- TEST SET EVALUATION ON OLD TRAINING DATA ---")
    test_metrics, test_scores, test_labels = run_split_evaluation(df_test, "Test")
    #test_metrics, test_scores, test_labels = run_split_evaluation(df_test_old, "Test")
    print(test_metrics)
    save_metrics(test_metrics)

    plot_score_distribution(
        test_scores,
        test_labels,
        "test",
        "Test Score Distribution",
        #f"{fig_dir}/test_scores_{int(time.time())}.png",
    )
    plot_roc_curve(
    test_labels,
    test_scores,
    "old data test",
    )