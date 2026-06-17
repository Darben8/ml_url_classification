## Project Overview

This repository implements a multi-signal phishing URL classification pipeline. The current system is not an agentic workflow or orchestration framework. It is a sequential inference pipeline that gathers several signals for each URL and combines them through either fixed-score fusion or learned meta-model fusion.

The pipeline currently uses four primary signals:

- `bert_score`: a character-level BERT model score
- `cb_score` / `cb_benign_prob`: a CatBoost lexical-feature model score
- `vt_score` or VirusTotal-derived reputation features
- `tranco_score` and rank-derived popularity features

These signals support multiple fusion paths:

- standard average fusion
- weighted average fusion
- rich-feature stacking
- compact 4-signal stacking
- selected saved meta-model evaluation for model comparison and ablation experiments

## Current Architecture

At runtime, the pipeline is centered on `ml_inference()` in [graph/nodes/inference.py](/abs/c:/Users/Manubea/Documents/Career/Grad%20School/GRA/Research%20Project/ml_url_classification/graph/nodes/inference.py:17).

For each input URL, the pipeline:

1. normalizes the registered domain
2. looks up Tranco popularity
3. looks up VirusTotal reputation
4. runs character-level BERT inference
5. runs CatBoost inference
6. records signal-specific error flags and neutral fallbacks
7. computes average and weighted ensemble outputs
8. optionally computes stacking outputs with a saved meta model

The implementation is function-based and artifact-driven. The `graph/` package name is historical, but the behavior is a deterministic signal-generation and fusion pipeline rather than a graph-routed autonomous system.

## Signal Definitions

### Character-Level BERT

The BERT path converts a URL string into character IDs, feeds them into the saved BERT checkpoint, and returns a benign-style probability:

- higher `bert_score` means more benign
- lower `bert_score` means more phishing-like

Artifacts are stored under `data/bert_model/`, including alternative saved checkpoints such as cross-validation outputs.

### CatBoost Lexical Model

The CatBoost path extracts handcrafted lexical URL features such as:

- URL length
- number of dots
- digit usage
- suspicious keywords
- TLD indicators
- entropy and subdomain properties

It returns phishing and benign probabilities, with `cb_score` / `cb_benign_prob` used as the benign-style feature for fusion.

### VirusTotal Reputation

The VirusTotal path queries the URL reputation service and derives:

- `vt_detection_rate`
- `vt_malicious_count`
- `vt_suspicious_count`
- `vt_total_engines`

For simple fusion, the code converts detection rate into a benign-style score:

- `vt_score = 1 - vt_detection_rate`

VirusTotal responses are cached locally in SQLite to reduce repeated API calls.

### Tranco Popularity

The Tranco path extracts the registered domain and looks it up in a local top-1M ranking. It derives:

- `in_tranco`
- `tranco_rank`
- `tranco_score`

Higher-ranked and known-popular domains receive more benign-like scores.

## Label Convention

The repository uses the following internal label convention across inference, training, and evaluation:

- `Benign = 1`
- `Phishing = 0`

For score-like fields:

- values closer to `1` mean more benign / more trusted
- values closer to `0` mean more phishing-like / less trusted

This applies to:

- `bert_score`
- `cb_score`
- `cb_benign_prob`
- `vt_score`
- `tranco_score`
- `ensemble_score`
- `weighted_score`
- `stacking_score`

## Fusion Modes

### Standard Average Fusion

Average fusion uses the arithmetic mean of:

- `tranco_score`
- `vt_score`
- `bert_score`
- `cb_score`

It writes:

- `ensemble_score`
- `std_prediction`

### Weighted Average Fusion

Weighted fusion uses manually assigned weights in [graph/nodes/ensemble2.py](/abs/c:/Users/Manubea/Documents/Career/Grad%20School/GRA/Research%20Project/ml_url_classification/graph/nodes/ensemble2.py:1):

- Tranco: `0.15`
- VirusTotal: `0.35`
- BERT: `0.25`
- CatBoost: `0.25`

It writes:

- `weighted_score`
- `weighted_prediction`

Unlike the standard ensemble, the weighted path can return `Uncertain`.

### Stacking Fusion

The stacking path treats model outputs and reputation-derived values as features rather than fixed votes. The inference entry point is [graph/nodes/stacking_inference.py](/abs/c:/Users/Manubea/Documents/Career/Grad%20School/GRA/Research%20Project/ml_url_classification/graph/nodes/stacking_inference.py:1).

It supports two feature families:

- `rich`: a larger feature set including signal details and error flags
- `4signal`: a compact feature set using only the core four benign-style signals

The prediction API returns:

- `stacking_phishing_prob`
- `stacking_score`
- `stacking_prediction`

## Stacking Feature Families

### Rich Feature Set

The rich stacker uses signal-level and operational features assembled in `models/fusion_features.py`. The current rich feature family includes:

- `bert_score`
- `cb_benign_prob`
- `vt_detection_rate`
- `vt_malicious_count`
- `vt_suspicious_count`
- `vt_total_engines`
- `in_tranco`
- `tranco_score`
- `tranco_rank`
- `bert_error`
- `catboost_error`
- `vt_error`
- `tranco_error`

The default rich runtime directory in [models/meta_model.py](/abs/c:/Users/Manubea/Documents/Career/Grad%20School/GRA/Research%20Project/ml_url_classification/models/meta_model.py:5) is:

- `data/ml_models/meta_model_v2`

### Compact 4-Signal Feature Set

The compact stacker uses:

- `bert_score`
- `cb_score`
- `vt_score`
- `tranco_score`

This path is useful for simpler, faster learned fusion and for ablation-style comparisons that focus on the four highest-level signals.

The default 4-signal runtime directory in [models/meta_model.py](/abs/c:/Users/Manubea/Documents/Career/Grad%20School/GRA/Research%20Project/ml_url_classification/models/meta_model.py:5) is:

- `data/ml_models/meta_model_4signal_v1`

## Data Sources and Splits

The repository currently uses two labeled datasets with different roles.

### Newer Dataset

- `data/phishing_url_dataset_unique.csv`

This is the newer dataset used for sampling, splitting, stacker feature generation, and most current evaluation workflows.

From [graph/nodes/load_data.py](/abs/c:/Users/Manubea/Documents/Career/Grad%20School/GRA/Research%20Project/ml_url_classification/graph/nodes/load_data.py:1), it is sampled into `url_sample` and then split into:

- `df_dev`
- `df_val`
- `df_test`

### Older Dataset

- `data/new_data_urls.csv`

This older labeled dataset is still preserved for comparison workflows and separate evaluation splits:

- `df_dev_old`
- `df_val_old`
- `df_test_old`

### Intended Roles

Current intended usage is:

- `df_dev` from the newer dataset for stacker feature generation and stacker training
- `df_val` and `df_test` for evaluating current fusion behavior
- old-data splits for historical comparison

## Main Scripts

### `eval.py`

Legacy evaluation script for the earlier workflow. It still runs the current signal-generation pipeline but preserves the older evaluation style.

### `eval2.py`

Main configurable evaluation script for current fusion experiments.

It supports:

- `data_type = "new_data"` or `data_type = "old_data"`
- `fusion_mode = "average"`
- `fusion_mode = "stacking_rich"`
- `fusion_mode = "stacking_4signal"`
- `fusion_mode = "stacking_selected"`

For `stacking_selected`, the script can evaluate a specific saved meta-model directory and explicitly choose the feature family to match that artifact.

It writes consolidated metrics to:

- `data/results/results.csv`

It also generates score-distribution and ROC figures with filenames keyed by dataset, fusion mode, and BERT architecture.

### `train_stacker.py`

Trains the calibrated logistic-regression rich stacker used in the main learned-fusion path.

Current behavior:

- builds rich feature rows from `df_dev`
- saves feature rows to `data/results/stacker_training_features.csv`
- runs stratified cross-validation
- fits the final calibrated classifier
- saves artifacts to `data/ml_models/meta_model_v2`

Saved artifacts include:

- `logistic_regression_calibrated.pkl`
- `signal_feature_columns.pkl`
- `meta_model_metadata.json`

### `model_training/train_meta_models.py`

Trains multiple alternative meta-model families over a selected feature set. This script expands the project beyond the original calibrated logistic-regression stacker.

Supported feature-set families:

- `rich_signal`
- `4signal`

Supported model families include:

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- Gradient Boosting
- K Nearest Neighbors
- CatBoost
- Support Vector Machine
- Multi-layer Perceptron
- XGBoost, when installed
- LightGBM, when installed

This script writes model artifacts under `data/ml_models/` with names such as:

- `meta_model_4signal_gb_v1`
- `meta_model_rich_dt_v1`
- `meta_model_rich_lgbm_v1`

### `eval_meta_models.py`

Evaluates saved meta models against validation and test splits and appends comparison rows to `data/results/results.csv`.

It can evaluate:

- standard saved model directories
- chosen saved model directories
- both sources together

It automatically discovers compatible models, infers whether they are `stacking_rich` or `stacking_4signal`, and records both evaluation metrics and training metadata.

### `train_stacker_ablation.py`

Trains ablation-oriented meta models from selected subsets of the saved feature CSVs.

Supported ablation groups:

- `4sig`
- `rich`
- `richops`

The current ablation subsets include:

- `intrinsic_only`
- `intrinsic_and_vt`
- `intrinsic_and_tranco`
- `extrinsic`
- `all_signals`
- `bert_vt`
- `bert_tranco`
- `cb_vt`
- `cb_tranco`

Supported ablation model families include:

- Logistic Regression
- Support Vector Machine
- Multi-layer Perceptron
- Decision Tree
- Gradient Boosting
- CatBoost
- XGBoost

Artifacts are written under:

- `data/ml_models/ablation/`

This script also appends training summaries to:

- `data/results/all_model_train_results.csv`

## Meta-Model Selection State

The repository now contains multiple saved meta-model candidates rather than a single learned-fusion artifact.

Important pieces of the current selection workflow:

- `data/ml_models/active_meta_model.json` stores a selected-model record and selection rationale
- `eval_meta_models.py` compares saved candidates on evaluation splits
- `eval2.py` can directly point at a chosen saved model directory with `fusion_mode = "stacking_selected"`

The current `active_meta_model.json` records:

- selected meta-model id: `meta_model_4signal_gb_v1`
- feature set: `4signal`
- model family: `Gradient Boosting`
- selection basis: best accuracy / inference-time tradeoff from `data/results/results.csv`
- selection date: `2026-06-03`

Important note: the default runtime loader in `models/meta_model.py` still uses the hard-coded default directories for `rich` and `4signal` unless an explicit `model_dir` override is provided. In other words, the active-selection manifest documents a chosen model, but runtime behavior depends on which script and model directory are actually used.

## Evaluation Outputs

The repository contains multiple result artifacts, including:

- `data/results/results.csv`
- `data/results/all_model_train_results.csv`
- `data/results/stacker_training_features.csv`
- `data/results/stacker_training_features_4signal.csv`
- `data/results/paper_ablation_test.csv`
- `data/results/paper_ablation_test2.csv`
- `data/results/paper_ablation_smoketest.csv`
- `data/results/paper_runtime_analysis.csv`
- `data/results/paper_robustness.csv`
- `data/results/paper_error_analysis_false_positives.csv`
- `data/results/paper_error_analysis_false_negatives.csv`
- `data/results/paper_error_analysis_all.csv`

These outputs support:

- baseline versus stacking comparisons
- meta-model family comparison
- feature-family comparison
- ablation analysis
- runtime analysis
- robustness analysis
- error analysis

## Runtime and Paper-Oriented Analysis Scripts

The repository also includes:

- `paper_runtime_analysis.py`
- `paper_robustness.py`
- `paper_error_analysis.py`

These scripts build on the same underlying `ml_inference()` and stacking workflow to support paper-style evaluation and reporting.

## Required Artifacts and Prerequisites

To run the current pipeline successfully, the following are required.

Environment and secrets:

- Python environment with packages from `requirements.txt`
- `.env` file containing `VIRUSTOTAL_API_KEY`

Required data files:

- `data/tranco_top_1m.csv`
- `data/vt_cache.db` or `data/vt_cache_old.db`
- `data/phishing_url_dataset_unique.csv`
- `data/new_data_urls.csv`

Required base-model artifacts:

- BERT artifacts under `data/bert_model/`
- CatBoost artifacts under `data/ml_models/`

Required learned-fusion artifacts depend on the path being evaluated:

- standard rich stacker artifacts under `data/ml_models/meta_model_v2`
- standard 4-signal stacker artifacts under `data/ml_models/meta_model_4signal_v1`
- optional comparison and ablation artifacts under `data/ml_models/ablation/`, `data/ml_models/chosen_meta_models/`, and other `meta_model_*` directories

Operational notes:

- VirusTotal API calls dominate runtime when cache coverage is low
- uncached VirusTotal requests are rate-limited
- meta-model inference requires the feature schema in the artifact directory to match the feature builder used for that model

## Current Project State

The project has evolved beyond a single average-fusion baseline and a single calibrated logistic-regression stacker. It now includes:

- sequential multi-signal inference
- average and weighted score fusion
- rich and compact stacking feature families
- multiple trained meta-model families
- ablation-driven model training
- selected-model evaluation workflows
- paper-oriented runtime, robustness, and error-analysis scripts

The current repository is best understood as a practical phishing-detection experimentation pipeline for comparing signal combinations and learned fusion strategies over multiple saved model families.
