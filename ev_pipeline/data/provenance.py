"""Input hashes and descriptive evidence about the data."""
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ND, NH, Z
from ..features.calendar import calendar, dayidx
from ..scoring.metrics import truth
from ..utils.io import file_sha, save_json, write_table


EXPECTED_DATA_SHA256 = {
    "adj.csv": "aca00131f6e2b53c8c4e25741fd41f9e85e617940f49022495209b64e41074e6",
    "distance.csv": "77d04203982b55b222e9d470ce1736153afea1234d2e0ad76bbda5889307476d",
    "duration.csv": "38bfe12ccac44cb0ce0ebd901f1b8438fa0a6c603ef1020874d9a74902fabc08",
    "e_price.csv": "523ebf45697d4f0d47e8fc3623d7b9b6ba31bc40792d76898f51240ac9578bfd",
    "inf.csv": "d18fa98a426a30e76490ab5930a7880ba462bff2bcd6d5a6ccdbae810ee64b3d",
    "inf_raw.csv": "0db3bcaa77d2c87f2891462a5a0c01455356502880e1c2b6807bd13222bb80c0",
    "occupancy.csv": "c0542657b60f1b526cadd0f2e1a3413e52040ded2ef5c4ad0e8c5122ecd1a463",
    "poi.csv": "f28f09074ea661fac2c834576cd91d18b3e188bd26c034b41452f7777538d76e",
    "s_price.csv": "fcdb6659a08b53c73ca5ad34eb558896ffe52893c211f85f5da1c4c5ebd9ef69",
    "volume-11kW.csv": "055763e293279f44529ccf2e1b4f8ed394599ad23b39a8edf5c2b53f66d9a0c5",
    "volume.csv": "b4dd71781eb6a6ce6bded3dea6ea269cd99e20b1fa1e36133ab5e0aeb67ec78c",
    "weather_airport.csv": "7852dc463859d462b359785575e352651bb7bea63b5c6b0f11eed65256f6faf8",
    "weather_central.csv": "fc3e0a7b062495f8e6c3e0fed887f7dea5e0f49cdf3fa4698a15a82e45c92ce9",
}


def save_data_provenance(data, out):
    actual = {name: file_sha(Path(data) / name) for name in EXPECTED_DATA_SHA256}
    same = actual == EXPECTED_DATA_SHA256
    save_json(
        Path(out) / "data_provenance.json",
        dict(
            input_sha256=actual,
            equivalent_snapshot="44f2aa0c8d89f192bce00bafb0def74a21b39c68" if same else None,
            snapshot_data_match=same,
        ),
    )


def descriptive_evidence(D, out):
    V = D["V"].reshape(ND, 24, Z)
    daily = V.sum((1, 2))
    cal = calendar()
    cc = D["st"].cc.to_numpy(float)
    capped = pd.read_csv(Path(D["data_path"]) / "volume-11kW.csv", index_col=0).to_numpy(float)
    tet = dayidx("2023-01-21")
    normal = (cal.dow == cal.dow.iloc[tet]) & (cal.hol == 0) & (cal.mk == 0)
    audit = dict(
        days=ND,
        hours=NH,
        zones=Z,
        stations=int(D["st"].nst.sum()),
        piles=int(cc.sum()),
        volume_total=float(V.sum()),
        volume11_total=float(capped.sum()),
        volume11_ratio=float(capped.sum() / V.sum()),
        zero_share=float((V == 0).mean()),
        daily_mean=float(daily.mean()),
        daily_min=float(daily.min()),
        daily_max=float(daily.max()),
        daily_median=float(np.median(daily)),
        spring_festival_day_volume=float(daily[tet]),
        spring_drop_vs_nonholiday_same_weekday=float(1 - daily[tet] / daily[normal].mean()),
        same_weekday_comparison_days=int(normal.sum()),
        max_occupancy_ratio=float((D["O"] / cc).max()),
        label="Descriptive; includes the final test window",
    )
    ranked = np.sort(V.sum((0, 1)))[::-1]
    for k in (10, 27, 28, 69):
        audit[f"top{k}_share"] = float(ranked[:k].sum() / ranked.sum())
    counts = 0
    types = set()
    for chunk in pd.read_csv(
        Path(D["data_path"]) / "poi.csv", usecols=["primary_types"], chunksize=100000
    ):
        counts += len(chunk)
        types.update(chunk.primary_types.unique())
    audit.update(poi_points=counts, poi_types=sorted(types))
    save_json(Path(out) / "data_audit.json", audit)
    write_table(
        out,
        "daily_city",
        pd.DataFrame(
            dict(
                date=cal.index.astype(str),
                volume_kwh=daily,
                off=cal.off.values,
                holiday=cal.hol.values,
                makeup=cal.mk.values,
            )
        ),
    )
    write_table(
        out,
        "hourly_profile",
        pd.DataFrame(
            dict(
                hour=range(24),
                workday=V.sum(-1)[cal.off == 0].mean(0),
                restday=V.sum(-1)[cal.off == 1].mean(0),
                all_days=V.sum(-1).mean(0),
            )
        ),
    )
    rows = []
    for target in ("volume", "occ"):
        daily_target = truth(D, target).mean(1)
        scale = np.array([daily_target[d - 7 : d].mean(0) for d in range(7, ND)])
        original = 1e-3 * np.median(scale)
        training = 1e-3 * np.median(scale[:134])
        rows.append(
            dict(
                target=target,
                original_floor=original,
                training_floor=training,
                relative_difference=original / training - 1,
                binding_zone_days=int((scale < original).sum()),
                final_binding_zone_days=int((scale[134:] < original).sum()),
                eligible_zone_days=scale.size,
            )
        )
    write_table(out, "floor_binding_summary", rows)
