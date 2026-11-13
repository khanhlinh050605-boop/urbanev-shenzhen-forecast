"""Read the raw CSV panels into one in-memory dictionary."""
from pathlib import Path

import pandas as pd


def load_data(data):
    data = Path(data)
    rd = lambda name: pd.read_csv(data / (name + ".csv"), index_col=0, parse_dates=True)
    V, O, E, S, W = (
        rd("volume"),
        rd("occupancy"),
        rd("e_price"),
        rd("s_price"),
        rd("weather_central"),
    )
    inf = pd.read_csv(data / "inf.csv")
    g = inf.groupby("TAZID")
    st = pd.DataFrame(
        {
            "cc": g.charge_count.sum(),
            "nst": g.size(),
            "area": g.area.first(),
            "perim": g.perimeter.first(),
        }
    )
    st.index = st.index.astype(str)
    zones = list(V.columns)
    st = st.loc[zones]
    adj = pd.read_csv(data / "adj.csv").values.astype(float)
    return dict(
        V=V.values,
        O=O.values,
        E=E.values,
        S=S.values,
        W=W[["T", "U", "nRAIN"]].values.astype(float),
        st=st,
        adj=adj,
        zones=zones,
        index=V.index,
        inf=inf,
        data_path=data,
        frames=dict(volume=V, occupancy=O, e_price=E, s_price=S, weather=W),
    )
