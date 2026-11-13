"""Lag, neighbour, weather and calendar feature construction and training-only scaling."""
import numpy as np

from ..config import LAGS, MIN_DAY, ND, NH, SEED, Z
from .calendar import calendar


def sh(A, L):
    o = np.full_like(A, np.nan)
    o[L:] = A[:-L]
    return o


def build_features(D, target="volume", noisy_fc=False, pert_from=None):
    cc = D["st"]["cc"].values.astype(float)
    Y0 = D["V"] if target == "volume" else D["O"] / cc
    X0 = D["O"] / cc if target == "volume" else D["V"]
    Y, X2, E, S, Wt = (Y0.copy(), X0.copy(), D["E"].copy(), D["S"].copy(), D["W"].copy())
    if pert_from is not None:
        r = np.random.default_rng(1)
        for A in (Y, X2, E, S, Wt):
            A[pert_from:] = r.random(A[pert_from:].shape) * 100
    cal = calendar()
    A3 = lambda A: A.reshape(ND, 24, -1)
    cols = {}
    dm = A3(Y).mean(1)
    dmax = A3(Y).max(1)
    Wn = D["adj"].copy()
    np.fill_diagonal(Wn, 0)
    rs = Wn.sum(1, keepdims=True)
    Wn = np.divide(Wn, rs, out=np.zeros_like(Wn), where=rs > 0)

    def dshift(A, k):
        o = np.full_like(A, np.nan)
        o[k:] = A[:-k]
        return o

    B = lambda A: np.broadcast_to(A[:, None, :], (ND, 24, A.shape[-1]))
    own = {}
    for L in LAGS:
        own[f"l{L}"] = A3(sh(Y, L))
    for k in (1, 2, 3):
        c = np.full((ND, Z), np.nan)
        c[1:] = A3(Y)[:-1, 24 - k, :]
        own[f"cut{k - 1}"] = B(c)
    own["dm1"] = B(dshift(dm, 1))
    own["dm_same"] = B(dshift(dm, 7))
    own["dmax1"] = B(dshift(dmax, 1))
    dm7 = np.full((ND, Z), np.nan)
    for e in range(7, ND):
        dm7[e] = dm[e - 7 : e].mean(0)
    own["dm7"] = B(dm7)
    lagstack = np.stack([own[f"l{L}"] for L in LAGS], 0)
    own["hr7"] = lagstack.mean(0)
    own["hr7std"] = lagstack.std(0)
    hd = np.full((ND, 24, Z), np.nan)
    Y3 = A3(Y)
    for e in range(7, ND):
        idx = np.arange(e - 7, -1, -7)
        hd[e] = Y3[idx].mean(0)
    own["hd_mean"] = hd
    xs = {}
    xs["x_l24"] = A3(sh(X2, 24))
    xs["x_l168"] = A3(sh(X2, 168))
    xdm = A3(X2).mean(1)
    xs["x_dm1"] = B(dshift(xdm, 1))
    xdm7 = np.full((ND, Z), np.nan)
    for e in range(7, ND):
        xdm7[e] = xdm[e - 7 : e].mean(0)
    nb, city = ({}, {})
    for k in ("l24", "l168", "cut0", "dm1", "hr7"):
        A = own[k]
        nb["nb_" + k] = A @ Wn.T
        city["city_" + k] = np.broadcast_to(A.mean(-1, keepdims=True), A.shape)
    T, U, R = [A3(Wt[:, [i]])[:, :, 0] for i in range(3)]
    Tm, Um, Rx = (T.mean(1), U.mean(1), R.max(1))
    Tm7 = np.full(ND, np.nan)
    for e in range(7, ND):
        Tm7[e] = Tm[e - 7 : e].mean()
    bz = lambda a: np.broadcast_to(a[:, :, None], (ND, 24, Z))
    wd = lambda a: bz(np.broadcast_to(dshift(a[:, None], 1), (ND, 24)))
    wx = {
        "w_T_l24": bz(A3(sh(Wt[:, [0]], 24))[:, :, 0]),
        "w_U_l24": bz(A3(sh(Wt[:, [1]], 24))[:, :, 0]),
        "w_R_l24": bz(A3(sh(Wt[:, [2]], 24))[:, :, 0]),
        "w_T_dm1": wd(Tm),
        "w_U_dm1": wd(Um),
        "w_R_dmax1": wd(Rx),
        "w_T_dm7": bz(np.broadcast_to(Tm7[:, None], (ND, 24))),
        "w_T_dev": wd(Tm) - bz(np.broadcast_to(Tm7[:, None], (ND, 24))),
    }
    pr = {"e_l24": A3(sh(E, 24)), "s_l24": A3(sh(S, 24))}
    cb = lambda a: np.broadcast_to(a[:, None, None], (ND, 24, Z))
    cs = {
        "hour": np.broadcast_to(np.arange(24)[None, :, None], (ND, 24, Z)),
        "dow": cb(cal.dow.values),
        "wk": cb(cal.wk.values),
        "hol": cb(cal.hol.values),
        "mk": cb(cal.mk.values),
        "off": cb(cal.off.values),
        "hpos": cb(cal.hpos.values),
        "hlen": cb(cal.hlen.values),
        "to_next": cb(cal.to_next.values),
        "since_end": cb(cal.since_end.values),
        "off_T": cb(np.r_[0, cal.off.values[:-1]]),
        "hol_T": cb(np.r_[0, cal.hol.values[:-1]]),
    }
    st = D["st"]
    stat = {
        c: np.broadcast_to(st[c].values.astype(float)[None, None, :], (ND, 24, Z))
        for c in ["cc", "nst", "area", "perim"]
    }
    stat["zone"] = np.broadcast_to(np.arange(Z)[None, None, :], (ND, 24, Z))
    groups = dict(own=own, xs=xs, nb=nb, city=city, wx=wx, pr=pr, cs=cs, stat=stat)
    if noisy_fc:
        r = np.random.default_rng(SEED)
        fT = D["W"][:, 0] + r.normal(0, 1.5, NH)
        fU = D["W"][:, 1] + r.normal(0, 8, NH)
        groups["fc"] = {"fc_T": bz(A3(fT[:, None])[:, :, 0]), "fc_U": bz(A3(fU[:, None])[:, :, 0])}
    mask_rows = np.arange(ND) >= MIN_DAY
    s = np.maximum(dm7, 0.001 * np.nanmedian(dm7))
    sB = B(s)
    xs_s = np.maximum(xdm7, 0.001 * np.nanmedian(xdm7))
    xsB = B(xs_s)
    cs_s = np.broadcast_to(np.nanmean(s, 1)[:, None, None], (ND, 24, Z))
    return dict(groups=groups, Y=A3(Y), s=sB, xs_s=xsB, city_s=cs_s, cal=cal, rows=mask_rows)


def build_split(D, target, training_end, mode="training_floor", pert_from=None):
    B = build_features(D, target, pert_from=pert_from)
    if mode == "original":
        return B
    cc = D["st"]["cc"].to_numpy(float)
    Y = D["V"] if target == "volume" else D["O"] / cc
    X = D["O"] / cc if target == "volume" else D["V"]
    if pert_from is not None:
        rng = np.random.default_rng(1)
        Y = Y.copy(order="K")
        X = X.copy(order="K")
        Y[pert_from:] = rng.random(Y[pert_from:].shape) * 100
        X[pert_from:] = rng.random(X[pert_from:].shape) * 100
    metadata = {}

    def scaling(A, name):
        dm = A.reshape(ND, 24, Z).mean(1)
        trailing = np.full_like(dm, np.nan)
        for day in range(7, ND):
            trailing[day] = dm[day - 7 : day].mean(0)
        floor = max(
            float(0.001 * np.nanmedian(trailing[MIN_DAY : training_end + 1])), np.finfo(float).tiny
        )
        full = float(0.001 * np.nanmedian(trailing))
        metadata[name] = dict(
            training_floor=floor,
            full_series_floor=full,
            training_end_day=training_end,
            training_binding_zone_days=int((trailing[7 : training_end + 1] < floor).sum()),
            all_binding_zone_days=int((trailing[7:] < floor).sum()),
            valid_zone_days=int(np.isfinite(trailing[7:]).sum()),
        )
        return np.broadcast_to(np.maximum(trailing, floor)[:, None, :], (ND, 24, Z))

    B["s"], B["xs_s"] = scaling(Y, "target"), scaling(X, "cross_target")
    B["city_s"] = np.broadcast_to(B["s"].mean(-1, keepdims=True), (ND, 24, Z))
    B["floor_meta"] = metadata
    return B
