"""Flatten the feature groups into a model matrix."""
import numpy as np

from ..config import LAGS, Z


OWN_LEVEL = [f"l{L}" for L in LAGS] + [
    "cut0",
    "cut1",
    "cut2",
    "dm1",
    "dm_same",
    "dmax1",
    "dm7",
    "hr7",
    "hr7std",
    "hd_mean",
]


def to_matrix(B, variant="base", ratio=False, fc=False):
    g = B["groups"]
    parts = {}
    for k, a in g["own"].items():
        if ratio and k == "dm7":
            continue
        parts[k] = a / B["s"] if ratio and k in OWN_LEVEL else a
    for k, a in g["xs"].items():
        parts[k] = a / B["xs_s"] if ratio else a
    if ratio:
        parts["log_s"] = np.log(B["s"])
    if variant == "nbr":
        for k, a in g["nb"].items():
            parts[k] = a / B["s"] if ratio else a
        for k, a in g["city"].items():
            parts[k] = a / B["city_s"] if ratio else a
    for grp in ("wx", "pr", "cs", "stat"):
        parts.update(g[grp])
    if fc:
        parts.update(g["fc"])
    names = list(parts)
    X = np.stack([np.asarray(parts[n], dtype=np.float32).reshape(-1) for n in names], 1)
    return (X, names)


def row_index(d0, d1):
    return slice(d0 * 24 * Z, (d1 + 1) * 24 * Z)
