import torch
from models.tokenizer import load_char_to_idx
from models.bert_architecture import create_attention_mask

#best bert fold from cross validation



char_to_idx = load_char_to_idx()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def url_to_tensor(url: str, max_len: int = 200):
    """
    Converts a URL string to a tensor of character indices for Char-BERT.

    Args:
        url (str): The URL to convert.
        max_len (int): Maximum sequence length (same as model).

    Returns:
        input_ids: Tensor of shape [1, max_len]
        attention_mask: Tensor of shape [1, max_len]
    """
    # Map characters to indices; unknown characters -> 0
    indices = [char_to_idx.get(c, 0) for c in url]

    # Truncate or pad sequence
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices = indices + [0] * (max_len - len(indices))

    # Convert to tensor
    input_ids = torch.tensor([indices], dtype=torch.long, device=device)
    attention_mask = create_attention_mask(input_ids)
    #attention_mask = (input_ids != 0).long()


    return input_ids, attention_mask
