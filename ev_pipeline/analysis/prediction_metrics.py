"""Score saved predictions against the baselines."""
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FOLDS, HOLIDAYS, ND, QUANTILES, START, Z
from ..features.calendar import dayidx
from ..scoring.bootstrap import boot_ci
from ..scoring.metrics import Evaluator
from ..scoring.splits import evaluation
from ..utils.io import write_table


def prediction_inventory(out):
    files = []
    for split in ("val", "test"):
        for target in ("volume", "occ"):
            files.extend(
                Path(out) / split / target / (name + ".npy")
                for name in [
                    "snaive24",
                    "snaive168",
                    "persist_last",
                    "hour_dow_mean",
                    "mean7_samehour",
                    "ridge_AR",
                    "lgb_base",
                    "lgb_nbr",
                    "lgb_nbr_fcnoisy",
                    "lgb_nbr_ratio",
                    "xgb",
                    "lgb_xgb_blend",
                ]
            )
            files.extend(
                Path(out) / "extra" / "floor" / target / f"{name}_{split}.npy"
                for name in ["lgb", "xgb", "blend", "scale"]
            )
            files.extend(
                Path(out) / "extra" / "quant" / target / f"q{alpha:.8f}_{split}.npy"
                for alpha, _ in QUANTILES[target]
            )
    return files


def evaluate_predictions(D, out, seed):
    metrics_rows = []
    daily_rows = []
    paired = []
    zone_rows = []
    window_rows = []
    for target in ("volume", "occ"):
        parts = {}
        for split in ("val", "test"):
            ev = evaluation(D, target, split)
            predictions = {
                p.stem: np.load(p)
                for p in sorted((Path(out) / split / target).glob("*.npy"))
                if p.stem != "scale"
                and not p.stem.startswith("curves_")
                and not p.stem.startswith("curves_")
            }
            for model, name in [
                ("lgb_training_floor", "lgb"),
                ("xgb_training_floor", "xgb"),
                ("blend_training_floor", "blend"),
            ]:
                predictions[model] = np.load(
                    Path(out) / "extra" / "floor" / target / f"{name}_{split}.npy"
                )
            parts[split] = (ev, predictions)
            baseline = ev.metrics(predictions["snaive24"])
            for name, p in predictions.items():
                row = ev.metrics(p)
                row.update(
                    zh_bias=float((p - ev.A).sum() / np.abs(ev.A).sum()),
                    skill_zh_MAE_vs_snaive24=1 - row["zh_MAE"] / baseline["zh_MAE"],
                    skill_city_MAE_vs_snaive24=1 - row["city_MAE"] / baseline["city_MAE"],
                )
                metrics_rows.append(
                    dict(
                        target=target,
                        split=split,
                        model=name,
                        **row,
                        label="final test" if split == "test" else "validation",
                    )
                )
                dz, dc = ev.daily(p)
                for day, z, c in zip(ev.days, dz, dc):
                    daily_rows.append(
                        dict(
                            target=target,
                            split=split,
                            model=name,
                            day=int(day),
                            date=str((START + pd.Timedelta(days=int(day))).date()),
                            zone_MAE=float(z),
                            city_MAE=float(c),
                            label="final test" if split == "test" else "validation",
                        )
                    )
                ae = np.abs(p - ev.A)
                naive_ae = np.abs(predictions["snaive24"] - ev.A)
                for j, zone in enumerate(D["zones"]):
                    zone_rows.append(
                        dict(
                            target=target,
                            split=split,
                            model=name,
                            zone=zone,
                            MAE=float(ae[:, :, j].mean()),
                            WAPE=(
                                float(ae[:, :, j].sum() / np.abs(ev.A[:, :, j]).sum())
                                if np.abs(ev.A[:, :, j]).sum() > 0
                                else np.nan
                            ),
                            skill=(
                                float(1 - ae[:, :, j].sum() / naive_ae[:, :, j].sum())
                                if naive_ae[:, :, j].sum() > 0
                                else np.nan
                            ),
                            volume=float(D["V"].reshape(ND, 24, Z)[ev.days, :, j].sum()),
                            label="final test" if split == "test" else "validation",
                        )
                    )
                masks = (
                    [
                        (f"fold{i+1}", (ev.days >= a) & (ev.days <= b))
                        for i, (a, b) in enumerate(FOLDS)
                    ]
                    if split == "val"
                    else [("retrospective_test", np.ones(len(ev.days), bool))]
                )
                masks.extend(
                    (holiday, np.isin(ev.days, np.arange(dayidx(a), dayidx(b) + 1)))
                    for holiday, (a, b) in HOLIDAYS.items()
                )
                for window, mask in masks:
                    if mask.any():
                        wm = ev.metrics(p, mask)
                        bm = ev.metrics(predictions["snaive24"], mask)
                        window_rows.append(
                            dict(
                                target=target,
                                split=split,
                                model=name,
                                window=window,
                                **wm,
                                zone_MAE_skill=1 - wm["zh_MAE"] / bm["zh_MAE"],
                                city_MAE_skill=1 - wm["city_MAE"] / bm["city_MAE"],
                                snaive24_zh_WAPE=bm["zh_WAPE"],
                                snaive24_city_WAPE=bm["city_WAPE"],
                                label="final test" if split == "test" else "validation",
                            )
                        )
        ev = Evaluator(
            D,
            target,
            np.r_[parts["val"][0].days, parts["test"][0].days],
            np.r_[parts["val"][0].fe, parts["test"][0].fe],
        )
        models = set(parts["val"][1]) & set(parts["test"][1])
        pred = {
            name: np.concatenate([parts["val"][1][name], parts["test"][1][name]]) for name in models
        }
        parts["all_five_windows"] = (ev, pred)
        for split, (ev, pred) in parts.items():
            losses = {name: ev.daily(p) for name, p in pred.items()}
            pairs = {
                (name, ref) for ref in ["snaive24", "ridge_AR"] for name in pred if name != ref
            }
            pairs.add(("lgb_xgb_blend", "lgb_nbr_ratio" if target == "volume" else "xgb"))
            pairs.add(("blend_training_floor", "lgb_xgb_blend"))
            pairs.add(
                (
                    "blend_training_floor",
                    "lgb_training_floor" if target == "volume" else "xgb_training_floor",
                )
            )
            for model, reference in sorted(pairs):
                for level, i in [("zone_hour", 0), ("city_hour", 1)]:
                    for block in (1, 7):
                        mean, lo, hi, prob = boot_ci(
                            losses[model][i] - losses[reference][i], block=block, seed=seed
                        )
                        paired.append(
                            dict(
                                target=target,
                                split=split,
                                model=model,
                                reference=reference,
                                level=level,
                                block_days=block,
                                difference=mean,
                                lower=lo,
                                upper=hi,
                                bootstrap_fraction_below_zero=prob,
                                label="final test" if split != "val" else "validation",
                            )
                        )
    write_table(out, "all_model_metrics", metrics_rows)
    write_table(out, "daily_losses", daily_rows)
    write_table(out, "paired_confidence_intervals", paired)
    write_table(out, "zone_errors", zone_rows)
    write_table(out, "window_summary", window_rows)
