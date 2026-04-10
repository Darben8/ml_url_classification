## Project overview
This repository implements a URL phishing classifier that has evolved from an earlier "agentic/LangGraph" idea into a practical multi-signal phishing detection pipeline.

For each URL, the system gathers four main signals:

Char-level BERT score

A custom character-level BERT classifier converts the URL string into character IDs, runs inference, and outputs a benign probability.

CatBoost score

A CatBoost model extracts handcrafted lexical URL features such as length, dots, digits, suspicious keywords, TLD indicators, entropy, and subdomain properties, then outputs phishing/benign probabilities.

VirusTotal reputation score

The URL is looked up via the VirusTotal API. The code derives a detection rate from malicious/suspicious verdict counts, then converts that into a benign-style score via `1 - vt_detection_rate`. Results are cached locally in SQLite.

Tranco popularity score

The registered domain is extracted from the URL and looked up in a local Tranco top-1M CSV. The rank is normalized into a 0-1 trust-like score, where highly ranked domains score closer to 1.

These signals are currently used in two fusion styles:

Standard average fusion

A simple arithmetic mean across the four benign-style scores.

Stacking fusion

A calibrated logistic regression meta-model learns how to combine signal-level features derived from BERT, CatBoost, VirusTotal, and Tranco.

## Label and score convention
This project uses the following internal convention throughout inference, evaluation, and stacker training:

- `Benign = 1`
- `Phishing = 0`

For score-like outputs:

- values closer to `1` mean more benign / more trusted
- values closer to `0` mean more phishing-like / less trusted

This applies to:

- `bert_score`
- `cb_benign_prob` / `cb_score`
- `vt_score`
- `tranco_score`
- `ensemble_score`
- `stacking_score`

## Inference flow
At runtime, `ml_inference()` acts as the core signal-generation pipeline:

take a URL from input state,

normalize it to a registered domain,

query Tranco,

query VirusTotal,

run BERT inference,

run CatBoost inference,

compute standard ensemble decision,

compute weighted ensemble decision,

return all intermediate signals, error flags, and final predictions in a state dictionary.

The stacker path builds on top of this signal-generation step. It uses the signal outputs from `ml_inference()` as input features for a calibrated logistic regression model rather than relying only on fixed-score averaging.

## Data and evaluation
The repo includes:

labeled URL datasets,

trained BERT and CatBoost artifacts,

evaluation outputs and figures,

`eval.py`, which preserves the older evaluation workflow,

`eval2.py`, which supports configurable evaluation over multiple data types and fusion modes,

`train_stacker.py`, which builds signal-level training features and trains the calibrated logistic regression stacker.

Evaluation outputs include metrics such as accuracy, precision, recall, F1, ROC-AUC, and inference time, along with score-distribution and ROC-curve figures.

## New stacking pipeline
The repository now includes a learned-fusion path in addition to the standard averaging baseline.

The stacking pipeline works by treating model and reputation outputs as features rather than votes. The current stacker feature set is:

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

These features are assembled in `models/fusion_features.py`, passed through a calibrated logistic regression model, and returned as:

- `stacking_score`
- `stacking_prediction`

The current stacker is calibrated with sigmoid calibration and is designed to operate alongside the average ensemble rather than replacing it outright.

## Average Fusion Vs Stacking Fusion
Average fusion uses a fixed rule:

- compute the benign-style scores from BERT, CatBoost, VirusTotal, and Tranco
- take the arithmetic mean
- classify using a fixed threshold

Stacking fusion uses a learned rule:

- compute the same underlying signals
- flatten selected signal-level features into a feature vector
- feed the vector into a calibrated logistic regression model
- classify using the model's learned probability output

In short, average fusion is hand-defined and static, while stacking fusion is data-driven and trainable.

## eval2.py
`eval2.py` is the main configurable evaluation script for the current workflow.

It supports two data types:

- `new_data`
- `old_data`

These map to the following split sets from `graph/nodes/load_data.py`:

- `new_data`: `df_dev`, `df_val`, `df_test`
- `old_data`: `df_dev_old`, `df_val_old`, `df_test_old`

It also supports two fusion modes:

- `average`
- `stacking`

Behavior:

- `fusion_mode = "average"` evaluates the standard ensemble using `ensemble_score` and `std_prediction`
- `fusion_mode = "stacking"` evaluates the calibrated meta-model using `stacking_score` and `stacking_prediction`

`eval2.py` writes consolidated metrics to:

- `data/results/results.csv`

The output CSV includes:

- the original metric columns used by the earlier evaluation workflow
- `data_type`
- `bert_architecture`
- `fusion_mode`

Figure filenames and titles are also generated dynamically from the active data type, fusion mode, and BERT architecture.

## train_stacker.py
`train_stacker.py` trains the calibrated logistic regression meta-model used by the stacking pipeline.

Current training source:

- `data/phishing_url_dataset_unique.csv -> url_sample -> df_dev`

The script:

- loads the current dev split used for stacker training
- runs `ml_inference()` for each URL
- builds signal-level fusion features
- evaluates stacker performance with stratified cross-validation
- fits the final calibrated logistic regression model
- saves artifacts for later inference

Primary outputs:

- `data/results/stacker_training_features.csv`
- `data/ml_models/meta_model/logistic_regression_calibrated.pkl`
- `data/ml_models/meta_model/signal_feature_columns.pkl`
- `data/ml_models/meta_model/meta_model_metadata.json`

The script now includes:

- a `tqdm` progress bar during feature building
- explicit phase logging for loading data, building features, running CV, fitting, and saving artifacts

## Training and evaluation data roles
The repository currently uses two labeled datasets with different roles.

`data/new_data_urls.csv`

- represents an older labeled dataset used in prior base-model training work
- should not be used directly as stacker training data for the current BERT/CatBoost models if those models were trained on it

`data/phishing_url_dataset_unique.csv`

- represents the newer labeled dataset used in current sampling, splitting, and evaluation logic
- is sampled into `url_sample`
- is then split into `df_dev`, `df_val`, and `df_test`

Current intended roles:

- `df_dev` from `phishing_url_dataset_unique.csv` is used to train the stacker
- `df_val` and `df_test` are used to evaluate current fusion behavior
- `old_data` splits remain available for comparison against earlier data sources

This keeps the stacker training workflow better separated from the earlier base-model training data.

## Required artifacts and runtime prerequisites
To run the current pipeline successfully, the following are required.

Environment and secrets:

- a Python environment with the packages from `requirements.txt`
- a `.env` file containing `VIRUSTOTAL_API_KEY`

Required data files:

- `data/tranco_top_1m.csv`
- `data/vt_cache.db` (created/updated automatically for VT caching)
- `data/phishing_url_dataset_unique.csv`
- `data/new_data_urls.csv`

Required model artifacts:

- BERT checkpoint/config/character map under `data/bert_model/`
- CatBoost model artifacts under `data/ml_models/`

Required stacker artifacts for `fusion_mode = "stacking"`:

- `data/ml_models/meta_model/logistic_regression_calibrated.pkl`
- `data/ml_models/meta_model/signal_feature_columns.pkl`
- `data/ml_models/meta_model/meta_model_metadata.json`

Operational notes:

- VirusTotal requests can dominate runtime if cache coverage is low because the code rate-limits uncached API calls
- the stacker path depends on the saved meta-model artifacts being trained on the same feature schema expected by `models/fusion_features.py`

## Current architectural reality
Although the repo still contains some historical references to LangGraph and "agents," the implemented system is now a sequential signal-generation and fusion pipeline rather than a true agentic orchestration framework.

The current architecture is best understood as:

- signal extraction with BERT, CatBoost, VirusTotal, and Tranco
- fusion through either standard averaging or calibrated logistic-regression stacking
- evaluation through configurable scripts that compare fusion modes across dataset splits

The core processing is function-based, artifact-driven, and model-centric. There is no visible graph routing policy, agent planner, or multi-step autonomous state machine in the current implementation.

The project now includes:

- traditional score aggregation for baseline comparison
- a learned stacker for calibrated fusion
- explicit error flags and neutral fallback handling for unavailable signals
- consolidated evaluation outputs through `eval2.py`
