import json
from pathlib import Path

import torch
from models.bert_architecture import  CharBERTClassifier
from models.tokenizer import load_char_to_idx

ACTIVE_BERT_DIR = Path("data/bert_model/bert_cv3_f3")
ACTIVE_BERT_CHECKPOINT = ACTIVE_BERT_DIR / "bert_checkpoint.pt"
ACTIVE_BERT_CONFIG = ACTIVE_BERT_DIR / "config.json"

_MODEL = None


def get_active_bert_metadata() -> dict:
    config = {}
    if ACTIVE_BERT_CONFIG.exists():
        with ACTIVE_BERT_CONFIG.open("r", encoding="utf-8") as f:
            config = json.load(f)

    hidden_size = config.get("hidden_size", "unknown")
    num_hidden_layers = config.get("num_hidden_layers", "unknown")
    num_attention_heads = config.get("num_attention_heads", "unknown")

    architecture_label = (
        f"{ACTIVE_BERT_DIR.name}_h{hidden_size}_"
        f"l{num_hidden_layers}_a{num_attention_heads}"
    )

    return {
        "model_dir": str(ACTIVE_BERT_DIR),
        "checkpoint_path": str(ACTIVE_BERT_CHECKPOINT),
        "config_path": str(ACTIVE_BERT_CONFIG),
        "bert_architecture": architecture_label,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
    }


def load_bert_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load character vocabulary
    char_to_idx = load_char_to_idx()
    vocab_size = len(char_to_idx) + 1  # +1 for padding/index 0

    # 2. Rebuild model architecture
    model = CharBERTClassifier(vocab_size=vocab_size)

    # 3. Load saved weights from checkpoint
    # checkpoint = torch.load(
    #     "data/bert_model/bert_checkpoint.pt",
    #     map_location=device
    # )

    #checkpoint from the best bert fold in cross validation
    #"data/bert_model/bert_crossval_best/bert_checkpoint.pt",
    checkpoint = torch.load(ACTIVE_BERT_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 4. Attach char_to_idx mapping
    model.char_to_idx = char_to_idx

    # 5. Move model to device and set eval mode
    model.to(device)
    model.eval()

    _MODEL = model
    return _MODEL




#with bert_architecture.py file
# _MODEL = None
# def load_bert_model(checkpoint_path="data/bert_model/bert_checkpoint.pt",
#                     config_path="data/bert_model/config.json",
#                     char_map_path="data/bert_model/char_to_idx.json"):
    
#     global _MODEL
#     if _MODEL is not None:
#         return _MODEL

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 1. Load config
#     config = BertConfig.from_json_file(config_path)

#     # 2. Build model from config
#     model = BertForSequenceClassification(config)
#     model.to(device)

#     # 3. Load checkpoint
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"])

#     # 4. Load char_to_idx mapping
#     with open(char_map_path, "r") as f:
#         model.char_to_idx = json.load(f)

#     model.eval()
#     _MODEL = model
#     return _MODEL
