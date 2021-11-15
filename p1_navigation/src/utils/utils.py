from hydra import compose, initialize

def get_seed() -> int:
    with initialize(config_path="conf"):
        return compose(config_name="config").seed

