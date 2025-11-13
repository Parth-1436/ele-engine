"""Minimal torch compatibility layer for ELE tests."""

import random as _random


def manual_seed(seed: int) -> None:
    """Mimic torch.manual_seed by seeding Python's random module."""

    _random.seed(seed)


__all__ = ["manual_seed"]
