"""Integrity checks on the input panels."""
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NH, START, Z
from ..utils.io import file_sha, save_json


def clean_data(D, data, out):
    frames = D["frames"]
    V = frames["volume"]
    O = frames["occupancy"]
    W = frames["weather"]
    checks = {}
    checks["volume_shape"] = V.shape == (NH, Z)
    checks["time_axis"] = bool(V.index.equals(pd.date_range(START, periods=NH, freq="h")))
    checks["aligned_panels"] = all(
        f.shape == (NH, Z) and f.index.equals(V.index) and list(f.columns) == list(V.columns)
        for name, f in frames.items()
        if name != "weather"
    )
    extra = {
        name: pd.read_csv(Path(data) / (name + ".csv"), index_col=0, parse_dates=True)
        for name in ["duration"]
    }
    checks["duration_alignment"] = all(
        f.shape == (NH, Z) and f.index.equals(V.index) and list(f.columns) == list(V.columns)
        for f in extra.values()
    )
    checks["finite_inputs"] = all(
        np.isfinite(a).all() for a in [D["V"], D["O"], D["E"], D["S"], D["W"]]
    )
    checks["no_missing_volume_occupancy"] = not V.isna().any().any() and not O.isna().any().any()
    checks["nonnegative_activity"] = bool((D["V"] >= 0).all() and (D["O"] >= 0).all())
    checks["zero_share"] = abs((D["V"] == 0).mean() - 0.0547) < 0.002
    checks["occupancy_range"] = float(D["O"].min()) == 0 and float(D["O"].max()) == 373
    daily = V.sum(axis=1).resample("D").sum()
    checks["daily_release_fingerprint"] = (
        abs(daily.mean() / 1.72e6 - 1) < 0.01
        and abs(daily.min() / 7.89e5 - 1) < 0.01
        and abs(daily.max() / 2.13e6 - 1) < 0.01
    )
    checks["weather_alignment"] = W.index.equals(V.index)
    checks["infrastructure"] = (
        len(D["inf"]) == 1362 and D["inf"].charge_count.sum() == 17532 and (D["st"].cc > 0).all()
    )
    checks["adjacency"] = D["adj"].shape == (Z, Z) and (D["adj"] != D["adj"].T).sum() // 2 == 20
    save_json(
        Path(out) / "data_checks.json",
        {
            "checks": {k: bool(v) for k, v in checks.items()},
            "all_pass": all(checks.values()),
            "input_sha256": {p.name: file_sha(p) for p in sorted(Path(data).glob("*.csv"))},
            "values_changed": False,
        },
    )
    if not all(checks.values()):
        raise ValueError("Input data checks failed")
    return D
