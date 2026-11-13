"""Zone-total normalisation benchmark."""
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.reproduction import PUBLISHED_TABLE3
from .upstream_sources import STATISTICAL_SOURCES
from ..utils.io import file_sha, save_json


ORDER = [
    "persist_last",
    "snaive24",
    "snaive168",
    "hour_dow_mean",
    "mean7_samehour",
    "ridge_AR",
    "lgb_base",
    "lgb_nbr",
    "lgb_nbr_ratio",
    "xgb",
    "lgb_xgb_blend",
    "lgb_nbr_fcnoisy",
]


def normalization_benchmark(data, out):
    import os
    from types import SimpleNamespace
    from statsmodels.tsa.ar_model import AutoReg
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        r2_score,
    )

    OUT = Path(out) / "published_baselines"
    OUT.mkdir(parents=True, exist_ok=True)
    ROOT = Path(out).parent
    DATA = Path(data)
    dump = lambda name, value: save_json(OUT / name, value)
    sha = file_sha
    od = OUT / "zone_total_diagnostic"
    od.mkdir(exist_ok=True)
    ns = dict(
        np=np,
        pd=pd,
        os=os,
        AutoReg=AutoReg,
        mean_squared_error=mean_squared_error,
        mean_absolute_error=mean_absolute_error,
        mean_absolute_percentage_error=mean_absolute_percentage_error,
        r2_score=r2_score,
    )
    hashes = {}
    for name, text in STATISTICAL_SOURCES.items():
        exec(text.replace("../data", str(DATA).replace("\\", "/")), ns)
    inf = pd.read_csv(DATA / "inf.csv")
    raw = pd.read_csv(DATA / "occupancy.csv", index_col=0)
    inf["zone"] = inf.TAZID.astype(str)
    last = inf.groupby("zone", sort=False).charge_count.last().reindex(raw.columns)
    total = inf.groupby("zone", sort=False).charge_count.sum().reindex(raw.columns)
    counts = inf.groupby("zone", sort=False).size().reindex(raw.columns)
    proof = pd.DataFrame(
        dict(
            zone=raw.columns,
            station_rows=counts.values,
            last_station_piles=last.values,
            zone_total_piles=total.values,
            denominator_ratio=(total / last).values,
        )
    )
    proof.to_csv(OUT / "pile_normalization_evidence.csv", index=False)
    actual_dict = dict(zip(inf.TAZID.astype(str), inf.charge_count))
    assert np.array_equal(last.values, np.array([actual_dict[c] for c in raw.columns]))
    feat = raw.to_numpy(dtype=float) / total.to_numpy(dtype=float)
    args = SimpleNamespace(
        seq_len=12,
        pred_len=3,
        total_fold=6,
        fold=1,
        feat="occ",
        pred_type="region",
        add_feat="None",
    )
    timeindex = pd.to_datetime(raw.index)

    def ar_fast(train, n):
        fit = AutoReg(train, lags=12).fit()
        b = np.asarray(fit.params)
        seq = list(train)
        out = []
        for _ in range(n):
            p = float(b[0] + np.dot(b[1:], np.asarray(seq[-12:][::-1])))
            out.append(p)
            seq.append(p)
        return np.asarray(out)

    dump(
        "diagnostic_provenance.json",
        dict(
            label="diagnostic variant, not the upstream setup",
            evaluation="retrospective authors short-horizon folds, separate from T+1",
            source_hashes=hashes,
            diagnostic_script_sha256=sha(__file__),
            data_hashes={name: sha(DATA / name) for name in ["occupancy.csv", "inf.csv"]},
            only_change="Divide raw occupancy by summed charge_count across all station rows per TAZID, rather than the last station charge_count retained by dict(zip()).",
            source_preserved="Exact upstream split_cv, Lo, Ar, metrics AST bodies; AR OLS recursion checked against exact refit loop before use",
            source_dictionary_last_row_verified_zones=len(raw.columns),
            zones_with_multiple_station_rows=int((counts > 1).sum()),
            zones_with_different_denominator=int((total != last).sum()),
            raw_occupancy_max=float(raw.to_numpy().max()),
            zone_total_normalized_max=float(feat.max()),
            folds=6,
            horizons=[3, 6, 9, 12],
            training_validation_test_fractions=[0.8, 0.1, 0.1],
        ),
    )
    checks = []
    rows = []
    for fold in range(1, 7):
        args.fold = fold
        tr, va, te, *_, scaler = ns["split_cv"](args, timeindex, feat, 0.8, 0.1, 0.1, "None")
        fit = np.vstack([tr, va, te[:12]])
        nmax = len(te) - 15
        for z in sorted(set([0, 1, int(np.argmax(feat.mean(0))), int(np.argmin(feat.mean(0)))])):
            exact = ns["Ar"](3, args, lags=12).predict(
                np.vstack([fit[:, [z]], np.zeros((3, 1))]), np.zeros((nmax, 1))
            )[:, 0]
            fast = ar_fast(fit[:, z], len(exact))
            err = float(np.max(np.abs(exact - fast)))
            checks.append(
                dict(
                    fold=fold,
                    zone_index=z,
                    forecast_hours=len(exact),
                    max_abs_difference=err,
                    passed=bool(np.allclose(exact, fast, rtol=1e-08, atol=1e-09)),
                )
            )
        dump("diagnostic_ar_equivalence_checks.json", checks)
        assert all((x["passed"] for x in checks))
        ar = np.column_stack([ar_fast(fit[:, z], nmax) for z in range(feat.shape[1])])
        for h in (3, 6, 9, 12):
            args.pred_len = h
            truth = te[12 + h :]
            tv = np.vstack([tr, va, te[: 12 + h]])
            for model, pred in dict(
                lo=ns["Lo"](args).predict(tv, truth), ar=ar[: len(truth)]
            ).items():
                assert np.isfinite(pred).all()
                vals = ns["metrics"](pred.copy(), truth.copy(), args)
                rows.append(
                    dict(
                        model=model,
                        fold=fold,
                        horizon_hours=h,
                        status="complete",
                        **dict(zip(["MSE", "RMSE", "MAPE", "RAE", "MAE"], vals)),
                    )
                )
                np.save(od / f"{model}_fold{fold}_h{h}.npy", pred.astype(np.float32))
            np.save(od / f"truth_fold{fold}_h{h}.npy", truth.astype(np.float32))
        pd.DataFrame(rows).to_csv(od / "results.csv", index=False)
        print("diagnostic fold", fold, "complete", flush=True)
    summary = (
        pd.DataFrame(rows)
        .groupby(["model", "horizon_hours"])[["RMSE", "MAPE", "RAE", "MAE"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(od / "summary.csv", index=False)
    old = pd.read_csv(Path(out) / "authors" / "lo_ar" / "summary.csv").rename(
        columns={m: "upstream_" + m for m in ["RMSE", "MAPE", "RAE", "MAE"]}
    )
    published = pd.DataFrame(PUBLISHED_TABLE3).rename(
        columns={m: "paper_" + m for m in ["RMSE", "MAPE", "RAE", "MAE"]}
    )
    comparison = (
        summary.rename(columns={m: "diagnostic_" + m for m in ["RMSE", "MAPE", "RAE", "MAE"]})
        .merge(old, on=["model", "horizon_hours"])
        .merge(published, on=["model", "horizon_hours"])
    )
    for metric in ("RMSE", "MAE"):
        comparison[metric + "_closer_to_paper"] = abs(
            comparison["diagnostic_" + metric] - comparison["paper_" + metric]
        ) < abs(comparison["upstream_" + metric] - comparison["paper_" + metric])
    comparison.to_csv(OUT / "normalization_diagnostic_comparison.csv", index=False)
    print("Completed one diagnostic variant, LO/AR, six folds, four horizons", flush=True)
