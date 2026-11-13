"""Reproduction of the published statistical tables."""
from pathlib import Path

import numpy as np
import pandas as pd

from ..benchmarks.upstream_sources import STATISTICAL_SOURCES
from ..utils.io import save_json


PUBLISHED_TABLE3 = [
    {
        "model": "lo",
        "horizon_hours": 3,
        "RMSE": 0.0975,
        "MAPE": 0.2539,
        "RAE": 0.3661999999999999,
        "MAE": 0.0491,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, LO, 3h; all displayed entries divided by 100",
    },
    {
        "model": "lo",
        "horizon_hours": 6,
        "RMSE": 0.1252,
        "MAPE": 0.3907,
        "RAE": 0.5405,
        "MAE": 0.0726,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, LO, 6h; all displayed entries divided by 100",
    },
    {
        "model": "lo",
        "horizon_hours": 9,
        "RMSE": 0.1465,
        "MAPE": 0.5092,
        "RAE": 0.6808,
        "MAE": 0.0917,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, LO, 9h; all displayed entries divided by 100",
    },
    {
        "model": "lo",
        "horizon_hours": 12,
        "RMSE": 0.1545,
        "MAPE": 0.5670000000000001,
        "RAE": 0.7404999999999999,
        "MAE": 0.0998,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, LO, 12h; all displayed entries divided by 100",
    },
    {
        "model": "ar",
        "horizon_hours": 3,
        "RMSE": 0.1308,
        "MAPE": 0.583,
        "RAE": 0.6754000000000001,
        "MAE": 0.0899,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, AR, 3h; all displayed entries divided by 100",
    },
    {
        "model": "ar",
        "horizon_hours": 6,
        "RMSE": 0.13,
        "MAPE": 0.5912,
        "RAE": 0.6707,
        "MAE": 0.0898,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, AR, 6h; all displayed entries divided by 100",
    },
    {
        "model": "ar",
        "horizon_hours": 9,
        "RMSE": 0.1289,
        "MAPE": 0.6093,
        "RAE": 0.6666,
        "MAE": 0.0896,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, AR, 9h; all displayed entries divided by 100",
    },
    {
        "model": "ar",
        "horizon_hours": 12,
        "RMSE": 0.1267,
        "MAPE": 0.6174000000000001,
        "RAE": 0.6565000000000001,
        "MAE": 0.0883,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, AR, 12h; all displayed entries divided by 100",
    },
    {
        "model": "arima",
        "horizon_hours": 3,
        "RMSE": 0.1376,
        "MAPE": 0.5863,
        "RAE": 0.7081999999999999,
        "MAE": 0.0946,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, ARIMA, 3h; all displayed entries divided by 100",
    },
    {
        "model": "arima",
        "horizon_hours": 6,
        "RMSE": 0.1388,
        "MAPE": 0.5889,
        "RAE": 0.7090000000000001,
        "MAE": 0.0951999999999999,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, ARIMA, 6h; all displayed entries divided by 100",
    },
    {
        "model": "arima",
        "horizon_hours": 9,
        "RMSE": 0.1344,
        "MAPE": 0.5956,
        "RAE": 0.6920999999999999,
        "MAE": 0.0931,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, ARIMA, 9h; all displayed entries divided by 100",
    },
    {
        "model": "arima",
        "horizon_hours": 12,
        "RMSE": 0.1278999999999999,
        "MAPE": 0.591,
        "RAE": 0.6578,
        "MAE": 0.0883999999999999,
        "source_url": "https://felix-rz.github.io/pdf/2025_UrbanEV.pdf",
        "locator": "Table 3 PDF page 7, ARIMA, 12h; all displayed entries divided by 100",
    },
]


def reproduction_details(D, out):
    out = Path(out)
    frame = pd.read_csv(out / "reproduction_comparison.csv")
    frame["percent_of_prior"] = 100 * frame.difference / frame.prior.replace(0, np.nan)
    precise = frame[
        (frame.split == "val")
        & frame.section.isin(["precise_pooled_MAE", "overall"])
        & (frame.metric == "zh_MAE")
    ]
    final = frame[
        (frame.split == "test") & (frame.section == "overall") & (frame.metric == "zh_MAE")
    ]
    city = frame[
        (frame.section == "overall") & frame.metric.isin(["city_MAE", "city_RMSE", "city_WAPE"])
    ]
    save_json(
        out / "reproduction_size_summary.json",
        dict(
            validation_max_abs_pooled_zone_mae_percent=float(precise.percent_of_prior.abs().max()),
            test_max_abs_pooled_zone_mae_percent=float(final.percent_of_prior.abs().max()),
            city_mae_rmse_wape_max_abs_percent=float(city.percent_of_prior.abs().max()),
            final_label="final test",
        ),
    )
    p = np.load(out / "test" / "occ" / "lgb_nbr.npy")
    b = np.load(out / "test" / "occ" / "snaive24.npy")
    bound = float(
        (
            (np.abs(np.spacing(p)).astype(float) + np.abs(np.spacing(b)).astype(float))
            * 0.5
            * D["st"].cc.to_numpy(float)
        )
        .sum(-1)
        .mean()
    )
    row = frame[
        (frame.target == "occ")
        & (frame.split == "test")
        & (frame.section == "bootstrap_vs_snaive24")
        & (frame.model == "lgb_nbr")
        & (frame.metric == "city_dMAE")
        & (frame.element == 0)
    ].iloc[0]
    save_json(
        out / "serialization_bound.json",
        dict(
            observed_difference=float(row.difference),
            tolerance=float(row.tolerance),
            prediction_storage_loss_bound=bound,
            consistent_with_storage_bound=bool(
                abs(row.difference) <= bound + 0.5 * row.tolerance / 1.1
            ),
            exact_cause_established=False,
            within_tolerance=bool(row.within_tolerance),
        ),
    )


def refresh_statistical_tables(out):
    import re
    import io
    from contextlib import redirect_stdout
    from types import SimpleNamespace
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        r2_score,
    )

    namespace = dict(
        np=np,
        pd=pd,
        mean_squared_error=mean_squared_error,
        mean_absolute_error=mean_absolute_error,
        mean_absolute_percentage_error=mean_absolute_percentage_error,
        r2_score=r2_score,
    )
    exec(STATISTICAL_SOURCES["utils.py"], namespace)
    out = Path(out)
    folders = [
        (out / "authors" / "lo_ar", ["lo", "ar"]),
        (out / "authors" / "arima", ["arima"]),
        (out / "published_baselines" / "zone_total_diagnostic", ["lo", "ar"]),
    ]
    for folder, models in folders:
        if not folder.exists():
            if models == ["arima"]:
                continue
            raise FileNotFoundError(folder)
        rows = []
        for actual_path in sorted(folder.glob("truth_fold*_h*.npy")):
            match = re.fullmatch(r"truth_fold(\d+)_h(\d+)", actual_path.stem)
            fold, horizon = map(int, match.groups())
            truth_array = np.load(actual_path).astype(float)
            args = SimpleNamespace(
                seq_len=12,
                pred_len=horizon,
                total_fold=6,
                fold=fold,
                feat="occ",
                pred_type="region",
                add_feat="None",
            )
            for model in models:
                path = folder / f"{model}_fold{fold}_h{horizon}.npy"
                if not path.exists():
                    raise FileNotFoundError(path)
                pred = np.load(path).astype(float)
                if np.isfinite(pred).all():
                    with redirect_stdout(io.StringIO()):
                        values = namespace["metrics"](pred.copy(), truth_array.copy(), args)
                    row = dict(zip(["MSE", "RMSE", "MAPE", "RAE", "MAE"], map(float, values)))
                    row["status"] = "complete"
                else:
                    row = dict(
                        status="failed_nonfinite", nonfinite_cells=int((~np.isfinite(pred)).sum())
                    )
                rows.append(dict(model=model, fold=fold, horizon_hours=horizon, **row))
        frame = pd.DataFrame(rows).reindex(
            columns=[
                "model",
                "fold",
                "horizon_hours",
                "status",
                "MSE",
                "RMSE",
                "MAPE",
                "RAE",
                "MAE",
                "nonfinite_cells",
            ]
        )
        if models == ["lo", "ar"] and len(frame) != 48:
            raise ValueError("Incomplete LO/AR benchmark cache")
        frame.to_csv(folder / "results.csv", index=False)
        frame.groupby(["model", "horizon_hours"])[
            ["RMSE", "MAPE", "RAE", "MAE"]
        ].mean().reset_index().to_csv(folder / "summary.csv", index=False)
    normal = pd.read_csv(
        out / "published_baselines" / "zone_total_diagnostic" / "summary.csv"
    ).rename(columns={m: "diagnostic_" + m for m in ["RMSE", "MAPE", "RAE", "MAE"]})
    upstream = pd.read_csv(out / "authors" / "lo_ar" / "summary.csv").rename(
        columns={m: "upstream_" + m for m in ["RMSE", "MAPE", "RAE", "MAE"]}
    )
    paper = pd.DataFrame(PUBLISHED_TABLE3).rename(
        columns={m: "paper_" + m for m in ["RMSE", "MAPE", "RAE", "MAE"]}
    )
    comparison = normal.merge(upstream, on=["model", "horizon_hours"]).merge(
        paper, on=["model", "horizon_hours"]
    )
    for metric in ("RMSE", "MAE"):
        comparison[metric + "_closer_to_paper"] = abs(
            comparison["diagnostic_" + metric] - comparison["paper_" + metric]
        ) < abs(comparison["upstream_" + metric] - comparison["paper_" + metric])
    comparison.to_csv(
        out / "published_baselines" / "normalization_diagnostic_comparison.csv", index=False
    )
