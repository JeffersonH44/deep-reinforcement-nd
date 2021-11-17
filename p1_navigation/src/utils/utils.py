import torch
import numpy as np
import math

from hydra import compose

def get_seed() -> int:
    return compose(config_name="config").seed

def get_device():
    device = compose(config_name="config").train_device
    if device == "gpu" and torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

# taken from here https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
def get_next_power_of_two(value):
    return int(math.pow(2, math.ceil(math.log(value)/math.log(2))))