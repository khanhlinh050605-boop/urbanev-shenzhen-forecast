"""Validation folds and the final-test window."""
import numpy as np

from ..config import FOLDS, MIN_DAY
from .metrics import Evaluator


def split_specs(split):
    return (
        [(MIN_DAY, a - 1, a, b) for a, b in FOLDS] if split == "val" else [(MIN_DAY, 140, 141, 180)]
    )


def evaluation(D, target, split):
    specs = split_specs(split)
    days = np.concatenate([np.arange(a, b + 1) for _, _, a, b in specs])
    ends = np.concatenate([np.full(b - a + 1, t1) for _, t1, a, b in specs])
    return Evaluator(D, target, days, ends)
