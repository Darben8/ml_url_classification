## Project overview
This repository implements a URL phishing classifier that has evolved from an earlier “agentic/LangGraph” idea into a more practical ensemble scoring pipeline.

For each URL, the system gathers four main signals:

Char-level BERT score

A custom character-level BERT classifier converts the URL string into character IDs, runs inference, and outputs a benign probability.

CatBoost score

A CatBoost model extracts handcrafted lexical URL features such as length, dots, digits, suspicious keywords, TLD indicators, entropy, and subdomain properties, then outputs phishing/benign probabilities.

VirusTotal reputation score

The URL is looked up via the VirusTotal API. The code derives a detection rate from malicious/suspicious verdict counts, then converts that into a benign-style score via 1 - vt_detection_rate. Results are cached locally in SQLite.

Tranco popularity score

The registered domain is extracted from the URL and looked up in a local Tranco top-1M CSV. The rank is normalized into a 0–1 trust-like score, where highly ranked domains score closer to 1. 

These four scores are then combined in two ways:

Standard ensemble: simple arithmetic mean.

Weighted ensemble: fixed weights

Tranco: 0.15

VirusTotal: 0.35

BERT: 0.25

CatBoost: 0.25
The weighted output can return Benign, Phishing, or Uncertain depending on score bands. 

## Inference flow
At runtime, ml_inference() acts as the core pipeline:

take a URL from input state,

normalize it to a registered domain,

query Tranco,

query VirusTotal,

run BERT inference,

run CatBoost inference,

compute standard ensemble decision,

compute weighted ensemble decision,

return all intermediate signals and final predictions in a state dictionary.

## Data and evaluation
The repo includes:

labeled URL datasets,

trained BERT and CatBoost artifacts,

evaluation outputs and figures,

a script (eval.py) that iterates through a validation/test split, runs the full ensemble on each URL, and writes metrics like accuracy, precision, recall, F1, ROC-AUC, and inference time.

## Current architectural reality
Although the repo still refers to LangGraph and “agents” in the README and docstrings, what is actually implemented now is mostly a single sequential scoring pipeline, not a real agentic orchestration system. The core processing is function-based, and there is no visible graph construction, routing policy, or multi-step state machine in the reviewed code.


URL Classification using BERT model, agentic orchestration with LangGraph.
Data used for this project include labelled dataset of phishing and benign urls and unlabelled dataset of the same.
Labelled was used for training BERT model, unlabelled for the agents.
All data is in the data folder.

Logic for the agents is in the services folder.
services/tranco.py handles retrieving ranking information from the Tranco database
services/virustotal.py handles retrieving url information from virus-total using the api

Preprocessing, tokenization of the BERT model as well as its architecture can be found in the models folder.
