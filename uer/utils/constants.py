import json


with open("models/special_tokens_map.json", mode="r", encoding="utf-8") as f:
    special_tokens_map = json.load(f)

UNK_TOKEN = special_tokens_map.get("unk_token")
CLS_TOKEN = special_tokens_map.get("cls_token")
SEP_TOKEN = special_tokens_map.get("sep_token")
MASK_TOKEN = special_tokens_map.get("mask_token")
PAD_TOKEN = special_tokens_map.get("pad_token")

# e.g. <extra_id_0>, <extra_id_1>, ... , should have consecutive IDs.
SENTINEL_TOKEN = special_tokens_map.get("sentinel_token")
