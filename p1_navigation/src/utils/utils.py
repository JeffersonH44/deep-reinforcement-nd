from hydra import compose

def get_seed() -> int:
    return compose(config_name="config").seed

