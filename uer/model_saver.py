import torch


def save_model(model, model_path):
    """
    Save model weights to file.
    """
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
