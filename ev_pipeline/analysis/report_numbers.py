"""Write every number quoted in the report's environment and investigation text to one table,
formatted as printed, so the text can be checked line by line against a single file."""
import json

import pandas as pd


def _ci(a, lo, hi, d=2):
    return f"{a:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def report_numbers(results):
    rows = []

    def add(section, item, value):
        rows.append(dict(section=section, item=item, value=value))

    prov = json.load(open(results / "execution_provenance.json"))
    add("environment", "python", prov["python"])
    for k, v in prov["packages"].items():
        add("environment", k, v)
    ex = json.load(open(results / "execution_summary.json"))
    add("runtime", "complete run, minutes", f"{ex['elapsed_seconds'] / 60:.0f}")
    fd = pd.read_csv(results / "authors" / "arima" / "fit_diagnostics.csv")
    add("runtime", "ARIMA fit time, minutes", f"{fd.elapsed_seconds.sum() / 60:.0f}")
    add("arima", "fits not converged", f"{int((~fd.converged.astype(bool)).sum()):,} of {len(fd):,}")
    s = pd.read_csv(results / "authors" / "arima" / "summary.csv").iloc[0]
    add("arima", "six-fold mean RMSE, MAE, MAPE, RAE", f"{s.RMSE:.3f}, {s.MAE:.3f}, {s.MAPE:.3f}, {s.RAE:.3f}")

    t = pd.read_csv(results / "zone_failure_tests.csv")
    names = {"zero_share": "zero-activity share", "festival_ratio": "festival-week ratio",
             "max_daily_jump_final": "largest daily jump", "test_vs_last30": "level vs last 30 days",
             "train_mean": "training level", "train_cv": "training variability"}
    for r in t.itertuples():
        add("zone failure AUC", f"{r.target}: {names.get(r.covariate, r.covariate)}",
            _ci(r.auc, r.auc_lower, r.auc_upper) + f"; median losing {r.median_losing:.2f} vs other {r.median_other:.2f}")
    if len(t):
        add("zone failure", "losing zones (energy, occupancy)",
            ", ".join(str(int(t[t.target == k].losing_zones.iloc[0])) for k in ("volume", "occ")))
    c = pd.read_csv(results / "zone_failure_concentration.csv")
    for r in c.itertuples():
        add("zone failure concentration", f"{r.target}: worst {r.top_losing_zones} zones", f"{100 * r.excess_share:.1f}%")
    a = pd.read_csv(results / "ablation_steps.csv")
    for r in a.itertuples():
        add("ablation", f"{r.target} {r.split}: {r.step}", _ci(r.percent, r.percent_lower, r.percent_upper, 1) + " %")
    g = results / "figure_feature_gain.csv"
    if g.exists():
        for r in pd.read_csv(g).itertuples():
            add("split gain", f"{r.target}: {r.feature_group}", f"{100 * r.gain_share:.1f}%")
    pd.DataFrame(rows).to_csv(results / "report_numbers.csv", index=False)
