import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
from models.tokenizer import load_char_to_idx

char_to_idx = load_char_to_idx()

vocab_size = len(char_to_idx) + 1
print("Inferred vocab size from checkpoint:", vocab_size)
# print(loaded_vocab_size)

#class for the original large 512/8 bert checkpoint before cross validation. This is the model architecture that should be used for evaluation since it corresponds to the best performing checkpoint.
class CharBERTClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=512,
        num_labels=2,
        max_len=200,
        num_hidden_layers=8,
        num_attention_heads=8
    ):
        super().__init__()

#class for the best fold in cross validation which has a different architecture than the original bert checkpoint. This is the model architecture that should be used for evaluation since it corresponds to the best performing checkpoint.
#for testing and will be changed
# class CharBERTClassifier(nn.Module):
#     def __init__(
#         self,
#         vocab_size=vocab_size,
#         hidden_size=128,
#         num_labels=2,
#         max_len=200,
#         num_hidden_layers=4,
#         num_attention_heads=4
#     ):
#         super().__init__()

        # Build a BERT config for character input
        self.config = BertConfig(
            vocab_size=vocab_size,          # your char vocabulary size
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_len,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            type_vocab_size=1,
            num_labels=num_labels
        )

        # Initialize BERT model
        self.bert = BertForSequenceClassification(self.config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

# Optional helper for preprocessing
def create_attention_mask(batch):
    return (batch != 0).long()
