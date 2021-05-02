import toml

from pathlib import Path


def load_config(config_path: Path) -> dict:
    config = {}
    with open(config_path) as f:
        # Read the parent config, like the rnn.toml
        config = toml.load(f)

    return config


def create_pipeline(list_functions):
    def pipeline(*input):
        res = input
        for function in list_functions:
            res = function(*res)
        return res

    return pipeline
