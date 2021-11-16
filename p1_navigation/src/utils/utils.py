import torch

from hydra import compose

def get_seed() -> int:
    return compose(config_name="config").seed

def get_device():
    device = compose(config_name="config").train_device
    if device == "gpu" and torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")