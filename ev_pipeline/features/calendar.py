"""Holiday / make-up-workday calendar features."""
import numpy as np
import pandas as pd

from ..config import HOLIDAYS, MAKEUP, ND, START


def dayidx(s):
    return (pd.Timestamp(s) - START).days


def calendar():
    d = pd.date_range(START, periods=ND)
    dow = d.dayofweek.values
    hol = np.zeros(ND, int)
    hpos = np.zeros(ND, int)
    hlen = np.zeros(ND, int)
    hid = np.zeros(ND, int)
    starts = []
    for k, (n, (a, b)) in enumerate(HOLIDAYS.items(), 1):
        i, j = (dayidx(a), dayidx(b))
        starts.append((i, j))
        hol[i : j + 1] = 1
        hpos[i : j + 1] = np.arange(1, j - i + 2)
        hlen[i : j + 1] = j - i + 1
        hid[i : j + 1] = k
    mk = np.zeros(ND, int)
    for a, b in MAKEUP:
        mk[dayidx(a) : dayidx(b) + 1] = 1
    wk = (dow >= 5).astype(int)
    off = (((wk == 1) | (hol == 1)) & (mk == 0)).astype(int)
    to_next = np.full(ND, 30)
    since_end = np.full(ND, 30)
    for e in range(ND):
        for i, j in starts:
            if i > e:
                to_next[e] = min(to_next[e], i - e)
            if j < e:
                since_end[e] = min(since_end[e], e - j)
    return pd.DataFrame(
        dict(
            dow=dow,
            wk=wk,
            hol=hol,
            mk=mk,
            off=off,
            hpos=hpos,
            hlen=hlen,
            to_next=np.minimum(to_next, 14),
            since_end=np.minimum(since_end, 14),
        ),
        index=d,
    )
