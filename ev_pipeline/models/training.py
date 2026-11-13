"""Main fitting loop that writes validation and test predictions."""
import gc
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..config import FOLDS, LOCKED, QUANTILES, TEST, Z
from ..features.engineering import build_features, build_split
from ..features.matrix import row_index, to_matrix
from .baselines import baseline_preds
from .gbm import fit_lgb_fixed, fit_xgb, lgb_params, predict_lgb, predict_xgb
from .ridge import ridge_inputs
from ..scoring.splits import split_specs
from ..utils.io import record_progress, save_array, save_json, write_table


def fit_predictions(D, out, seed):
    timing = []
    floors = []
    feature_names = None
    for target in ("volume", "occ"):
        settings = LOCKED[target]
        base_build = build_features(D, target)
        for split in ("val", "test"):
            specs = split_specs(split)
            accum = {}
            quantiles = {alpha: [] for alpha, _ in QUANTILES[target]}
            for t0, t1, a, b in specs:
                for name, p in baseline_preds(base_build, a, b).items():
                    accum.setdefault(("original", name), []).append(p)
                for mode in ("original", "training_floor"):
                    started = time.monotonic()
                    B = base_build if mode == "original" else build_split(D, target, t1)
                    X, names = to_matrix(B, "nbr", ratio=True)
                    feature_names = names
                    record_progress(
                        out,
                        phase="fitting",
                        target=target,
                        split=split,
                        mode=mode,
                        training_end=t1,
                        model="lightgbm",
                        label="final test" if split == "test" else "validation",
                    )
                    bst = fit_lgb_fixed(
                        X,
                        names,
                        B["Y"],
                        B["s"],
                        True,
                        (t0, t1),
                        lgb_params("l1", settings["leaves"], seed=seed),
                        settings["lgb_n"],
                    )
                    pl = predict_lgb(bst, X, B["s"], True, a, b, settings["lgb_n"])
                    gain = dict(zip(names, bst.feature_importance("gain").tolist()))
                    save_json(Path(out) / "importance" / f"{target}_{mode}_{a}_{b}.json", gain)
                    del bst
                    gc.collect()
                    record_progress(
                        out,
                        phase="fitting",
                        target=target,
                        split=split,
                        mode=mode,
                        training_end=t1,
                        model="xgboost",
                        label="final test" if split == "test" else "validation",
                    )
                    bst = fit_xgb(
                        X,
                        B["Y"],
                        B["s"],
                        True,
                        (t0, t1),
                        settings["xgb_n"],
                        depth=settings["depth"],
                        seed=seed,
                    )
                    px = predict_xgb(bst, X, B["s"], True, a, b, settings["xgb_n"])
                    del bst
                    gc.collect()
                    blend = settings["weight"] * pl + (1 - settings["weight"]) * px
                    if split == "val" and mode == "original":
                        blend = settings["weight"] * pl.astype(np.float32) + (
                            1 - settings["weight"]
                        ) * px.astype(np.float32)
                    for name, p in [
                        ("lgb", pl),
                        ("xgb", px),
                        ("blend", blend),
                        ("scale", B["s"][a : b + 1]),
                    ]:
                        accum.setdefault((mode, name), []).append(p)
                    if mode == "original":
                        xt = ridge_inputs(X, names, B["cal"], t0, t1)
                        scaler = StandardScaler().fit(xt)
                        rt = row_index(t0, t1)
                        rv = row_index(a, b)
                        y = B["Y"].reshape(-1)
                        s = B["s"].reshape(-1)
                        model = Ridge(alpha=settings["ridge_alpha"]).fit(
                            scaler.transform(xt), y[rt] / s[rt], sample_weight=s[rt]
                        )
                        p = (
                            model.predict(scaler.transform(ridge_inputs(X, names, B["cal"], a, b)))
                            * s[rv]
                        ).reshape(b - a + 1, 24, Z)
                        accum.setdefault(("original", "ridge_AR"), []).append(p)
                        del xt, scaler, model
                    else:
                        floors.append(
                            dict(
                                target=target, split=split, training_end=t1, floors=B["floor_meta"]
                            )
                        )
                        for alpha, n in QUANTILES[target]:
                            record_progress(
                                out,
                                phase="fitting",
                                target=target,
                                split=split,
                                mode=mode,
                                training_end=t1,
                                model="quantile",
                                alpha=alpha,
                                label="final test" if split == "test" else "validation",
                            )
                            params = lgb_params(
                                "l1",
                                settings["leaves"],
                                seed=seed,
                                objective="quantile",
                                metric="quantile",
                                alpha=alpha,
                            )
                            bst = fit_lgb_fixed(X, names, B["Y"], B["s"], True, (t0, t1), params, n)
                            quantiles[alpha].append(predict_lgb(bst, X, B["s"], True, a, b, n))
                            del bst
                            gc.collect()
                    timing.append(
                        dict(
                            target=target,
                            split=split,
                            mode=mode,
                            start_day=a,
                            end_day=b,
                            elapsed_seconds=time.monotonic() - started,
                            label="final test" if split == "test" else "validation",
                        )
                    )
                    del X, B, pl, px, blend
                    gc.collect()
            for (mode, name), parts in accum.items():
                if mode == "original":
                    alias = {"lgb": "lgb_nbr_ratio", "blend": "lgb_xgb_blend"}.get(name, name)
                    path = Path(out) / split / target / (alias + ".npy")
                else:
                    path = Path(out) / "extra" / "floor" / target / f"{name}_{split}.npy"
                save_array(path, np.concatenate(parts))
            qdir = Path(out) / "extra" / "quant" / target
            for alpha, parts in quantiles.items():
                save_array(qdir / f"q{alpha:.8f}_{split}.npy", np.concatenate(parts))
            save_json(
                qdir / "selection.json",
                dict(
                    quantiles={
                        f"q{alpha:.8f}": dict(alpha=alpha, n=n) for alpha, n in QUANTILES[target]
                    },
                    frozen=True,
                ),
            )
        del base_build
        gc.collect()
    write_table(out, "fit_runtimes", timing)
    save_json(Path(out) / "floor_constants.json", floors)
    save_json(
        Path(out) / "frozen_configuration.json",
        dict(
            models=LOCKED,
            quantiles=QUANTILES,
            folds=FOLDS,
            test=TEST,
            feature_names=feature_names,
            seed=seed,
        ),
    )


def fit_comparison_lightgbm(D, out, seed):
    settings = {"volume": {"leaves": 31, "n": 576}, "occ": {"leaves": 31, "n": 203}}
    times = []
    for target, cfg in settings.items():
        B = build_features(D, target)
        X, names = to_matrix(B, "base", ratio=False)
        for split in ("val", "test"):
            destination = Path(out) / split / target / "lgb_base.npy"
            predictions = []
            for t0, t1, a, b in split_specs(split):
                started = time.monotonic()
                bst = fit_lgb_fixed(
                    X,
                    names,
                    B["Y"],
                    B["s"],
                    False,
                    (t0, t1),
                    lgb_params("l1", cfg["leaves"], seed=seed),
                    cfg["n"],
                )
                predictions.append(predict_lgb(bst, X, B["s"], False, a, b, cfg["n"]))
                times.append(
                    dict(
                        target=target,
                        split=split,
                        model="lgb_base",
                        training_end=t1,
                        elapsed_seconds=time.monotonic() - started,
                        label="final test" if split == "test" else "validation",
                    )
                )
                del bst
                gc.collect()
            save_array(destination, np.concatenate(predictions))
        del B, X
        gc.collect()
    write_table(out, "comparison_fit_runtimes", times)
    save_json(Path(out) / "comparison_configuration.json", settings)
