"""Run settings and the fixed protocol constants (calendar, folds, locked model settings)."""
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42


ARIMA_TIME_LIMIT_SECONDS = None


ND, NH, Z = 181, 4344, 275


START = pd.Timestamp("2022-09-01")


SEED = RANDOM_SEED


LAGS = [24, 48, 72, 96, 120, 144, 168]


MIN_DAY = 7


FOLDS = [(30, 59), (60, 89), (90, 119), (120, 140)]


TEST = (141, 180)


HOLIDAYS = {
    "Mid_Autumn": ("2022-09-10", "2022-09-12"),
    "National_Day": ("2022-10-01", "2022-10-07"),
    "New_Year": ("2022-12-31", "2023-01-02"),
    "Spring_Festival": ("2023-01-21", "2023-01-27"),
}


MAKEUP = [("2022-10-08", "2022-10-09"), ("2023-01-28", "2023-01-29")]


LOCKED = {
    "volume": {
        "lgb_n": 160,
        "leaves": 31,
        "xgb_n": 79,
        "depth": 6,
        "weight": 0.75,
        "ridge_alpha": 100.0,
    },
    "occ": {
        "lgb_n": 102,
        "leaves": 127,
        "xgb_n": 250,
        "depth": 9,
        "weight": 0.35000000000000003,
        "ridge_alpha": 10000.0,
    },
}


QUANTILES = {
    "volume": [
        (0.1, 146),
        (0.2, 114),
        (1 / 3, 65),
        (0.5, 159),
        (2 / 3, 148),
        (0.75, 175),
        (0.8, 173),
        (0.9, 137),
    ],
    "occ": [(0.1, 99), (0.9, 76)],
}


RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0, 9.0)


@dataclass(frozen=True)
class Settings:
    data_path: Path = Path("data")
    output_path: Path = Path("output")
    seed: int = RANDOM_SEED
    reuse_saved_predictions: bool = False


def set_seed(seed):
    """Set the module-level seed. Must run before any other pipeline module is imported,
    because function defaults such as ``seed=SEED`` are bound at import time."""
    global RANDOM_SEED, SEED
    prefix = (__package__ or "") + "."
    late = [
        m
        for m in sys.modules
        if m.startswith(prefix) and m not in (prefix + "config", prefix + "cli", prefix + "__main__")
    ]
    if late and seed != SEED:
        raise RuntimeError("set_seed must run before importing pipeline modules: " + ", ".join(sorted(late)))
    RANDOM_SEED = SEED = seed
