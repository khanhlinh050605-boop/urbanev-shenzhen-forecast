"""Quantile calibration."""
from pathlib import Path

import numpy as np

from ..config import FOLDS, QUANTILES
from ..scoring.bootstrap import boot_ci
from ..scoring.splits import evaluation
from ..utils.io import save_array, save_json, write_table


def calibrate_quantiles(D, out, seed):
    rows = []
    crossings = []
    for target in ("volume", "occ"):
        qdir = Path(out) / "extra" / "quant" / target
        alphas = [a for a, _ in QUANTILES[target]]
        raw = {
            split: np.stack([np.load(qdir / f"q{a:.8f}_{split}.npy") for a in alphas])
            for split in ("val", "test")
        }
        fractions = {
            split: float(np.any(np.diff(p, axis=0) < 0, axis=0).mean()) for split, p in raw.items()
        }
        raw = {split: np.sort(p, axis=0) for split, p in raw.items()}
        ev = evaluation(D, target, "val")
        scale = np.load(Path(out) / "extra" / "floor" / target / "scale_val.npy")
        offsets = np.zeros(len(alphas))
        cal = raw["val"].copy()
        for a, b in FOLDS:
            hold = (ev.days >= a) & (ev.days <= b)
            past = ev.days < a
            if past.any():
                for i, alpha in enumerate(alphas):
                    cal[i, hold] += (
                        np.quantile(((ev.A - raw["val"][i]) / scale)[past], alpha) * scale[hold]
                    )
        for i, alpha in enumerate(alphas):
            offsets[i] = np.quantile((ev.A - raw["val"][i]) / scale, alpha)
        physical = lambda a: np.maximum(a, 0) if target == "volume" else np.clip(a, 0, 1)
        calibrated = {"val": np.sort(physical(cal), axis=0)}
        test_scale = np.load(Path(out) / "extra" / "floor" / target / "scale_test.npy")
        calibrated["test"] = np.sort(
            physical(raw["test"] + offsets[:, None, None, None] * test_scale[None]), axis=0
        )
        for split in ("val", "test"):
            save_array(qdir / f"calibrated_{split}.npy", calibrated[split])
            ev = evaluation(D, target, split)
            crossings.append(
                dict(
                    target=target,
                    split=split,
                    crossing_fraction=fractions[split],
                    label="final test" if split == "test" else "validation",
                )
            )
            for method, p in [
                ("raw", physical(raw[split])),
                ("forward_calibrated", calibrated[split]),
            ]:
                lo = p[alphas.index(0.1)]
                hi = p[alphas.index(0.9)]
                for level in ("zone_hour", "city_hour"):
                    actual = ev.A if level == "zone_hour" else (ev.A * ev.w).sum(-1)
                    lower = lo if level == "zone_hour" else (lo * ev.w).sum(-1)
                    upper = hi if level == "zone_hour" else (hi * ev.w).sum(-1)
                    covered = (actual >= lower) & (actual <= upper)
                    row = dict(
                        target=target,
                        split=split,
                        method=method,
                        level=level,
                        nominal_coverage=0.8,
                        coverage=float(covered.mean()),
                        mean_width=float((upper - lower).mean()),
                        lower_miss=float((actual < lower).mean()),
                        upper_miss=float((actual > upper).mean()),
                        label=(
                            "final test"
                            if split == "test"
                            else "forward validation calibration"
                        ),
                    )
                    if level == "zone_hour":
                        _, row["coverage_block7_lower"], row["coverage_block7_upper"], _ = boot_ci(
                            covered.mean((1, 2)), block=7, seed=seed
                        )
                    rows.append(row)
        save_json(
            qdir / "postprocessing.json",
            dict(
                validation_crossing_fraction=fractions["val"],
                test_crossing_fraction=fractions["test"],
                processing="monotone sorting and physical clipping",
            ),
        )
        save_json(
            qdir / "calibration.json",
            dict(
                alphas=alphas,
                offsets_scaled=offsets.tolist(),
                validation="earlier validation folds only; first fold raw",
                test="validation residuals only; final test",
            ),
        )
    write_table(out, "interval_metrics", rows)
    write_table(out, "quantile_crossing", crossings)
