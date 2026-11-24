import torch


_GPT_WORD_WEIGHT = "embedding.word.embedding.weight"
_GPT_POSITION_WEIGHT = "embedding.pos.embedding.weight"
_GPT_HEAD_WEIGHT = "target.lm.output_layer.weight"
_LEGACY_GPT_KEYS = {
    "embedding.word_embedding.weight": _GPT_WORD_WEIGHT,
    "embedding.position_embedding.weight": _GPT_POSITION_WEIGHT,
    "target.output_layer.weight": _GPT_HEAD_WEIGHT,
    "target.output_layer.bias": "target.lm.output_layer.bias",
}


def _upgrade_legacy_gpt_checkpoint(model, state_dict):
    # This layout belongs to causal GPT models, not every model with an LM head.
    if (not any(key in state_dict for key in _LEGACY_GPT_KEYS)
            or getattr(getattr(model, "encoder", None), "mask", None) != "causal"
            or getattr(model, "decoder", None) is not None
            or getattr(getattr(model, "target", None), "target_name_list", []) != ["lm"]):
        return

    model_weights = model.state_dict(keep_vars=True)
    if not all(key in model_weights for key in (_GPT_WORD_WEIGHT, _GPT_POSITION_WEIGHT,
                                                 _GPT_HEAD_WEIGHT)):
        return

    # Two names can address the same Parameter. A legacy alias must not overwrite
    # a current checkpoint value through the other name when the weights are tied.
    current_keys = set(state_dict)
    if model_weights[_GPT_WORD_WEIGHT] is model_weights[_GPT_HEAD_WEIGHT]:
        if _GPT_WORD_WEIGHT in current_keys or _GPT_HEAD_WEIGHT in current_keys:
            current_keys.update((_GPT_WORD_WEIGHT, _GPT_HEAD_WEIGHT))

    for old_key, new_key in _LEGACY_GPT_KEYS.items():
        if old_key in state_dict and new_key in model_weights:
            old_value = state_dict.pop(old_key)
            if new_key not in current_keys:
                state_dict[new_key] = old_value


def load_model(model, model_path):
    """Load model from saved weights."""
    state_dict = torch.load(model_path, map_location="cpu")
    unwrapped = model.module if hasattr(model, "module") else model
    _upgrade_legacy_gpt_checkpoint(unwrapped, state_dict)
    unwrapped.load_state_dict(state_dict, strict=False)
    return model
