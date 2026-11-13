"""Leakage and floor-scaling controls."""
import gc

import numpy as np

from ..features.engineering import build_features, build_split
from ..features.matrix import row_index, to_matrix
from ..utils.io import write_table


def leakage_checks(D, out):
    rows = []
    for target in ("volume", "occ"):
        for cutoff in (30, 60, 90, 100, 120, 141):
            before = build_split(D, target, cutoff - 1)
            after = build_split(D, target, cutoff - 1, pert_from=cutoff * 24)
            floor_equal = all(
                before["floor_meta"][key]["training_floor"]
                == after["floor_meta"][key]["training_floor"]
                for key in ["target", "cross_target"]
            )
            X, _ = to_matrix(before, "nbr", ratio=True)
            del before
            gc.collect()
            Y, _ = to_matrix(after, "nbr", ratio=True)
            del after
            gc.collect()
            r = row_index(7, cutoff)
            feature_equal = bool(np.array_equal(X[r], Y[r], equal_nan=True))
            rows.append(
                dict(
                    target=target,
                    cutoff_day=cutoff,
                    training_floor_unchanged=floor_equal,
                    ratio_features_unchanged=feature_equal,
                    passed=floor_equal and feature_equal,
                    label="held-out perturbation audit",
                )
            )
            del X, Y
            gc.collect()
    write_table(out, "leakage_checks", rows)
    if not all(r["passed"] for r in rows):
        raise RuntimeError("Feature availability checks failed")


def original_floor_controls(D, out):
    rows = []
    for target in ("volume", "occ"):
        for cutoff in (30, 60, 90, 100, 120, 141):
            first = build_features(D, target)
            changed = build_features(D, target, pert_from=cutoff * 24)
            ix = row_index(7, cutoff)
            x, _ = to_matrix(first, "nbr", ratio=True)
            y, _ = to_matrix(changed, "nbr", ratio=True)
            rows.append(
                dict(
                    target=target,
                    cutoff_day=cutoff,
                    ratio_features_unchanged=bool(np.array_equal(x[ix], y[ix], equal_nan=True)),
                )
            )
            del first, changed, x, y
            gc.collect()
    write_table(out, "original_floor_controls", rows)
