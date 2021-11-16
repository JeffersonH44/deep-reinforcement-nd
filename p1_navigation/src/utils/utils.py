import torch
import numpy as np

from hydra import compose

def get_seed() -> int:
    return compose(config_name="config").seed

def get_device():
    device = compose(config_name="config").train_device
    if device == "gpu" and torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def loop_choice(population, weights, k):
    # taken from here https://stackoverflow.com/questions/64135020/speed-up-random-weighted-choice-without-replacement-in-python
    wc = np.cumsum(weights)
    m = wc[-1]
    sample = np.empty(k, population.dtype)
    sample_idx = np.full(k, -1, np.int32)
    i = 0
    while i < k:
        r = m * np.random.rand()
        idx = np.searchsorted(wc, r, side='right')
        if np.isin(idx, sample_idx):
            continue
        sample[i] = population[idx]
        sample_idx[i] = population[idx]
        i += 1
    return sample