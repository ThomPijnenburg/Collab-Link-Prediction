import itertools
import numpy as np

from collections.abc import Callable
from random import randint


def sample_random(elements: list, n_samples: int) -> list:
    sampled_combinations = set()

    while len(sampled_combinations) < n_samples:
        # Choose one random item from each list; that forms an element

        elem = tuple([
            elements[randint(0, len(elements)-1)],
            elements[randint(0, len(elements)-1)]])
        # Using a set elminates duplicates easily
        sampled_combinations.add(elem)

    return list(sampled_combinations)


def create_random_sampler(n_samples_per_positive: int) -> Callable:
    def random_sampler(positive_batch) -> list:
        sample_nodes = list(itertools.chain(*positive_batch))
        negatives = sample_random(sample_nodes, n_samples_per_positive)
        return negatives

    return random_sampler


def create_sampler(method: str = "random", n_samples_per_positive: int = -1) -> Callable:
    def sampler(positive_batch) -> list:

        if method == "random":
            sampling_fn = create_random_sampler(n_samples_per_positive)

        return sampling_fn(positive_batch)

    return sampler
