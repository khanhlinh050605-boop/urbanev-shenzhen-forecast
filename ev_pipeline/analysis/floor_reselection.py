"""Hyper-parameter re-selection with the scale floor computed from each fold's training days only.

Stand-alone sensitivity check (not part of the default pipeline run):

    python -m ev_pipeline.analysis.floor_reselection --data <data> --out <out> [--floor training|original]

``--floor original`` repeats the selection with the full-series floor and should reproduce
``config.LOCKED``; ``--floor training`` uses ``build_split(D, target, fold_start - 1)`` so that no
floor ever sees a validation day. Only validation folds are used. The final test window is not touched.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..config import FOLDS, LOCKED, MIN_DAY, Z
from ..data.loader import load_data
from ..features.engineering import build_features, build_split
from ..features.matrix import row_index, to_matrix
from ..models.gbm import fit_xgb, lgb_params, predict_lgb, predict_xgb
from ..models.ridge import ridge_inputs

GRID_LGB = [(o, l) for o in ("l1", "l2") for l in (31, 127)]
GRID_XGB = (6, 9)
GRID_RIDGE = (1.0, 100.0, 10000.0)
GRID_W = np.round(np.linspace(0, 1, 21), 2)
DAYS = np.array([b - a + 1 for a, b in FOLDS], float)


def _log(*a):
    print(*a, flush=True)


def _fold_build(D, target, a, floor):
    return build_features(D, target) if floor == "original" else build_split(D, target, a - 1)


def _lgb_curve(X, names, B, tr, va, params, nmax):
    y, s = B["Y"].reshape(-1), B["s"].reshape(-1)
    rt, rv = row_index(*tr), row_index(*va)
    cat = [names.index("zone")]
    dtr = lgb.Dataset(X[rt], y[rt] / s[rt], weight=s[rt], categorical_feature=cat,
                      feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(X[rv], y[rv] / s[rv], weight=s[rv], reference=dtr, categorical_feature=cat,
                      free_raw_data=False)
    ev = {}
    bst = lgb.train(params, dtr, nmax, valid_sets=[dva], callbacks=[lgb.record_evaluation(ev)])
    return bst, np.array(ev["valid_0"]["l1"])


def _pick(curves):
    mc = (np.array(curves) * DAYS[:, None]).sum(0) / DAYS.sum()
    return int(mc.argmin()) + 1, float(mc.min())


def reselect(D, target, floor, nmax_l, nmax_x, seed):
    t0 = time.time()
    lg = {c: dict(b=[], cv=[]) for c in GRID_LGB}
    xg = {d: dict(b=[], cv=[]) for d in GRID_XGB}
    ridge = {al: [] for al in GRID_RIDGE}
    truth = []
    for a, b in FOLDS:  # pass 1: curves, boosters, ridge predictions
        B = _fold_build(D, target, a, floor)
        X, names = to_matrix(B, "nbr", ratio=True)
        truth.append(B["Y"][a : b + 1])
        for cfg in GRID_LGB:
            bst, cv = _lgb_curve(X, names, B, (MIN_DAY, a - 1), (a, b),
                                 lgb_params(cfg[0], cfg[1], seed=seed), nmax_l)
            lg[cfg]["b"].append(bst)
            lg[cfg]["cv"].append(cv)
            _log(f"[{target}/{floor}] fold {a}-{b} lgb {cfg} done {time.time() - t0:.0f}s")
        for d in GRID_XGB:
            bst, cv = fit_xgb(X, B["Y"], B["s"], True, (MIN_DAY, a - 1), nmax_x, depth=d,
                              seed=seed, va=(a, b))
            xg[d]["b"].append(bst)
            xg[d]["cv"].append(cv)
            _log(f"[{target}/{floor}] fold {a}-{b} xgb depth {d} done {time.time() - t0:.0f}s")
        xt = ridge_inputs(X, names, B["cal"], MIN_DAY, a - 1)
        sc = StandardScaler().fit(xt)
        y, s = B["Y"].reshape(-1), B["s"].reshape(-1)
        rt, rv = row_index(MIN_DAY, a - 1), row_index(a, b)
        xv = sc.transform(ridge_inputs(X, names, B["cal"], a, b))
        for al in GRID_RIDGE:
            m = Ridge(alpha=al).fit(sc.transform(xt), y[rt] / s[rt], sample_weight=s[rt])
            ridge[al].append((m.predict(xv) * s[rv]).reshape(b - a + 1, 24, Z))
        del X, B, xt, xv
        gc.collect()
    A = np.concatenate(truth)
    mae = lambda P: float(np.abs(np.concatenate(P) - A).mean())

    pick_l = {c: _pick(v["cv"]) for c, v in lg.items()}
    best_l = min(pick_l, key=lambda c: pick_l[c][1])
    pick_x = {d: _pick(v["cv"]) for d, v in xg.items()}
    best_x = min(pick_x, key=lambda d: pick_x[d][1])
    best_r = min(GRID_RIDGE, key=lambda al: mae(ridge[al]))

    # pass 2: predictions at the chosen and at the locked settings (same floors, same folds)
    lock = LOCKED[target]
    lock_cfg = ("l1", lock["leaves"])
    want = {"lgb_sel": (best_l, pick_l[best_l][0]), "lgb_lock": (lock_cfg, min(lock["lgb_n"], nmax_l))}
    want_x = {"xgb_sel": (best_x, pick_x[best_x][0]), "xgb_lock": (lock["depth"], min(lock["xgb_n"], nmax_x))}
    pred = {k: [] for k in list(want) + list(want_x)}
    for i, (a, b) in enumerate(FOLDS):
        B = _fold_build(D, target, a, floor)
        X, names = to_matrix(B, "nbr", ratio=True)
        for k, (cfg, n) in want.items():
            pred[k].append(predict_lgb(lg[cfg]["b"][i], X, B["s"], True, a, b, n))
        for k, (d, n) in want_x.items():
            pred[k].append(predict_xgb(xg[d]["b"][i], X, B["s"], True, a, b, n))
        del X, B
        gc.collect()
    P = {k: np.concatenate(v) for k, v in pred.items()}
    m_ = lambda p: float(np.abs(p - A).mean())
    sc_ = [m_(w * P["lgb_sel"] + (1 - w) * P["xgb_sel"]) for w in GRID_W]
    w_sel = float(GRID_W[int(np.argmin(sc_))])
    blend_sel = w_sel * P["lgb_sel"] + (1 - w_sel) * P["xgb_sel"]
    blend_lock = lock["weight"] * P["lgb_lock"] + (1 - lock["weight"]) * P["xgb_lock"]
    out = dict(
        target=target, floor=floor,
        selected=dict(lgb_obj=best_l[0], leaves=best_l[1], lgb_n=pick_l[best_l][0],
                      xgb_depth=best_x, xgb_n=pick_x[best_x][0], weight=w_sel, ridge_alpha=best_r),
        locked=dict(lgb_n=lock["lgb_n"], leaves=lock["leaves"], xgb_n=lock["xgb_n"],
                    depth=lock["depth"], weight=lock["weight"], ridge_alpha=lock["ridge_alpha"]),
        grid_lgb={f"{c[0]}_L{c[1]}": dict(n=v[0], curve_min=v[1]) for c, v in pick_l.items()},
        grid_xgb={f"depth{d}": dict(n=v[0], curve_min=v[1]) for d, v in pick_x.items()},
        pooled_val_zone_hour_mae=dict(
            blend_selected=m_(blend_sel), blend_locked=m_(blend_lock),
            lgb_selected=m_(P["lgb_sel"]), lgb_locked=m_(P["lgb_lock"]),
            xgb_selected=m_(P["xgb_sel"]), xgb_locked=m_(P["xgb_lock"]),
            ridge_selected=mae(ridge[best_r])),
        relative_change_blend_percent=100 * (m_(blend_sel) / m_(blend_lock) - 1),
        same_configuration=bool(
            best_l == lock_cfg and pick_l[best_l][0] == lock["lgb_n"] and best_x == lock["depth"]
            and pick_x[best_x][0] == lock["xgb_n"] and abs(w_sel - lock["weight"]) < 1e-9
            and best_r == lock["ridge_alpha"]),
        seconds=time.time() - t0,
    )
    _log(json.dumps(out["selected"]), "| locked:", json.dumps(out["locked"]))
    _log(f"[{target}/{floor}] blend MAE selected {out['pooled_val_zone_hour_mae']['blend_selected']:.6f} "
         f"vs locked {out['pooled_val_zone_hour_mae']['blend_locked']:.6f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", choices=("training", "original"), default="training")
    ap.add_argument("--targets", nargs="+", default=["volume", "occ"])
    ap.add_argument("--nmax-lgb", type=int, default=700)
    ap.add_argument("--nmax-xgb", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    D = load_data(a.data)
    res = Path(a.out) / "results"
    res.mkdir(parents=True, exist_ok=True)
    allres = [reselect(D, t, a.floor, a.nmax_lgb, a.nmax_xgb, a.seed) for t in a.targets]
    (res / f"floor_reselection_{a.floor}.json").write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
