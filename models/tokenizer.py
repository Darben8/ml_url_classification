import json

# def load_char_to_idx():
#     with open("data/bert_model/char_to_idx.json", "r") as f:
#         return json.load(f)
    

#load char_to_idx from the best bert fold in cross validation
def load_char_to_idx():
    #with open("data/bert_model/bert_crossval_best/char_to_idx.json", "r") as f:
    with open("data/bert_model/bert_cv3_f3/char_to_idx.json", "r") as f:
        return json.load(f)
    