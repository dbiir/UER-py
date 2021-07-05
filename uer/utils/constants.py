import json


with open("models/special_tokens_map.json", mode="r", encoding="utf-8") as f:
    special_tokens_map = json.load(f)

UNK_TOKEN = special_tokens_map["unk_token"]
CLS_TOKEN = special_tokens_map["cls_token"]
SEP_TOKEN = special_tokens_map["sep_token"]
MASK_TOKEN = special_tokens_map["mask_token"]
PAD_TOKEN = special_tokens_map["pad_token"]
try:
    SENTINEL_TOKEN = special_tokens_map["sentinel_token"]  # e.g. <extra_id_0>, <extra_id_1>, ... , should have consecutive IDs.
except KeyError:
    pass
