"""Additional fits (ridge, ablations and diagnostics)."""
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOCKED
from ..features.engineering import build_features, build_split
from ..features.matrix import to_matrix
from .baselines import baseline_preds
from .cluster import cluster_evidence
from .gbm import fit_lgb_fixed, fit_xgb, lgb_params, predict_lgb, predict_xgb
from ..scoring.bootstrap import boot_ci
from ..scoring.splits import evaluation, split_specs
from ..utils.io import record_progress, save_array, write_table


def extended_fits(D, out, seed, reuse):
    cluster_evidence(D, out, seed, reuse)
    out = Path(out)
    data11 = dict(D)
    data11["V"] = pd.read_csv(Path(D["data_path"]) / "volume-11kW.csv", index_col=0).to_numpy(float)
    for target in ("volume", "occ"):
        cfg = LOCKED[target]
        for split in ("val", "test"):
            if not reuse:
                for job in ("ratio_base", "seeds", "kw11"):
                    if job == "kw11" and target != "volume":
                        continue
                    accum = {}
                    DD = data11 if job == "kw11" else D
                    for t0, t1, a, b in split_specs(split):
                        B = build_split(DD, target, t1)
                        X, names = to_matrix(B, "nbr", ratio=True)
                        if job == "ratio_base":
                            keep = [i for i, n in enumerate(names) if not n.startswith("nb_")]
                            X = X[:, keep]
                            names = [names[i] for i in keep]
                        for ss in ((seed, 1, 2, 3, 4) if job == "seeds" else (seed,)):
                            record_progress(
                                out,
                                phase="extended_fits",
                                target=target,
                                split=split,
                                job=job,
                                seed=ss,
                                training_end=t1,
                            )
                            if job == "seeds" and ss == seed:
                                ix = slice(
                                    a - (30 if split == "val" else 141),
                                    b - (30 if split == "val" else 141) + 1,
                                )
                                pl = np.load(out / "extra" / "floor" / target / f"lgb_{split}.npy")[
                                    ix
                                ]
                                px = np.load(out / "extra" / "floor" / target / f"xgb_{split}.npy")[
                                    ix
                                ]
                            else:
                                bst = fit_lgb_fixed(
                                    X,
                                    names,
                                    B["Y"],
                                    B["s"],
                                    True,
                                    (t0, t1),
                                    lgb_params("l1", cfg["leaves"], seed=ss),
                                    cfg["lgb_n"],
                                )
                                pl = predict_lgb(bst, X, B["s"], True, a, b, cfg["lgb_n"])
                                del bst
                                if job != "ratio_base":
                                    bst = fit_xgb(
                                        X,
                                        B["Y"],
                                        B["s"],
                                        True,
                                        (t0, t1),
                                        cfg["xgb_n"],
                                        depth=cfg["depth"],
                                        seed=ss,
                                    )
                                    px = predict_xgb(bst, X, B["s"], True, a, b, cfg["xgb_n"])
                                    del bst
                            preds = (
                                {"lgb_ratio_no_neighbours": pl}
                                if job == "ratio_base"
                                else (
                                    {
                                        f"blend_seed{ss}": cfg["weight"] * pl
                                        + (1 - cfg["weight"]) * px
                                    }
                                    if job == "seeds"
                                    else {
                                        "blend": cfg["weight"] * pl + (1 - cfg["weight"]) * px,
                                        **baseline_preds(B, a, b),
                                    }
                                )
                            )
                            for name, pred in preds.items():
                                accum.setdefault(name, []).append(pred)
                        del B, X
                        gc.collect()
                    for name, parts in accum.items():
                        save_array(
                            out / "extra" / job / target / f"{name}_{split}.npy",
                            np.concatenate(parts),
                        )
                for model, trees in [
                    ("lgb_nbr", 677 if target == "volume" else 178),
                    ("lgb_nbr_fcnoisy", 655 if target == "volume" else 201),
                ]:
                    B = build_features(D, target, noisy_fc=model.endswith("fcnoisy"))
                    X, names = to_matrix(B, "nbr", ratio=False, fc=model.endswith("fcnoisy"))
                    parts = []
                    for t0, t1, a, b in split_specs(split):
                        record_progress(
                            out,
                            phase="diagnostic_fit",
                            target=target,
                            split=split,
                            model=model,
                            training_end=t1,
                        )
                        bst = fit_lgb_fixed(
                            X,
                            names,
                            B["Y"],
                            B["s"],
                            False,
                            (t0, t1),
                            lgb_params("l1", 31, seed=seed),
                            trees,
                        )
                        parts.append(predict_lgb(bst, X, B["s"], False, a, b, trees))
                        del bst
                    save_array(out / split / target / f"{model}.npy", np.concatenate(parts))
                    del B, X
                    gc.collect()
    rows = []
    seeds = []
    for target in ("volume", "occ"):
        for split in ("val", "test"):
            ev = evaluation(D, target, split)
            corrected = np.load(out / "extra" / "floor" / target / f"blend_{split}.npy")
            for ref, filename in [
                ("original_locked_blend", "lgb_xgb_blend"),
                ("snaive24", "snaive24"),
                ("ridge_AR", "ridge_AR"),
            ]:
                baseline = np.load(out / split / target / f"{filename}.npy")
                for level, idx in [("zone_hour", 0), ("city_hour", 1)]:
                    m, lo, hi, _ = boot_ci(
                        ev.daily(corrected)[idx] - ev.daily(baseline)[idx], block=7, seed=seed
                    )
                    rows.append(
                        dict(
                            experiment="training_floor",
                            target=target,
                            split=split,
                            reference=ref,
                            level=level,
                            difference=m,
                            lower=lo,
                            upper=hi,
                            **ev.metrics(corrected),
                        )
                    )
            no = np.load(
                out / "extra" / "ratio_base" / target / f"lgb_ratio_no_neighbours_{split}.npy"
            )
            yes = np.load(out / "extra" / "floor" / target / f"lgb_{split}.npy")
            m, lo, hi, _ = boot_ci(ev.daily(yes)[0] - ev.daily(no)[0], block=7, seed=seed)
            rows.append(
                dict(
                    experiment="neighbours_given_ratio",
                    target=target,
                    split=split,
                    reference="without_neighbours",
                    level="zone_hour",
                    difference=m,
                    lower=lo,
                    upper=hi,
                    without_WAPE=ev.metrics(no)["zh_WAPE"],
                    with_WAPE=ev.metrics(yes)["zh_WAPE"],
                )
            )
            for ss in (seed, 1, 2, 3, 4):
                pred = np.load(out / "extra" / "seeds" / target / f"blend_seed{ss}_{split}.npy")
                seeds.append(dict(target=target, split=split, seed=ss, **ev.metrics(pred)))
            if target == "volume":
                ev11 = evaluation(data11, target, split)
                pred = np.load(out / "extra" / "kw11" / target / f"blend_{split}.npy")
                base = np.load(out / "extra" / "kw11" / target / f"snaive24_{split}.npy")
                for level, idx in [("zone_hour", 0), ("city_hour", 1)]:
                    m, lo, hi, _ = boot_ci(
                        ev11.daily(pred)[idx] - ev11.daily(base)[idx], block=7, seed=seed
                    )
                    rows.append(
                        dict(
                            experiment="volume_11kw",
                            target=target,
                            split=split,
                            reference="snaive24",
                            level=level,
                            difference=m,
                            lower=lo,
                            upper=hi,
                            baseline_WAPE=ev11.metrics(base)["zh_WAPE"],
                            **ev11.metrics(pred),
                        )
                    )
    for row in rows:
        row["label"] = (
            "final test" if row["split"] == "test" else "validation sensitivity"
        )
    write_table(out, "sensitivity_summary", rows)
    write_table(out, "seed_metrics", seeds)
