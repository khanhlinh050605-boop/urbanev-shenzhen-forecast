"""LightGBM and XGBoost fit / predict wrappers."""
import lightgbm as lgb
import numpy as np
import xgboost as xgb

from ..config import SEED, Z
from ..features.matrix import row_index


def lgb_params(obj="l1", leaves=63, seed=SEED, **kw):
    p = dict(
        objective={"l1": "regression_l1", "l2": "regression"}[obj],
        metric="l1",
        learning_rate=0.08,
        num_leaves=leaves,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.7,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=127,
        num_threads=2,
        seed=seed,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
        cat_smooth=20,
        cat_l2=10,
    )
    p.update(kw)
    return p


def predict_lgb(bst, X, S, ratio, d0, d1, n):
    r = row_index(d0, d1)
    p = bst.predict(X[r], num_iteration=n)
    if ratio:
        p = p * S.reshape(-1)[r]
    return p.reshape(d1 - d0 + 1, 24, Z)


def fit_xgb(X, Y, S, ratio, tr, n, depth=7, seed=SEED, va=None):
    y = Y.reshape(-1)
    s = S.reshape(-1)
    rt = row_index(*tr)
    ytr = y[rt] / s[rt] if ratio else y[rt]
    dtr = xgb.DMatrix(X[rt], ytr, weight=s[rt] if ratio else None)
    p = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        eta=0.08,
        max_depth=depth,
        min_child_weight=40,
        subsample=0.7,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        max_bin=127,
        nthread=2,
        seed=seed,
    )
    ev = {}
    evals = []
    if va is not None:
        rv = row_index(*va)
        yv = y[rv] / s[rv] if ratio else y[rv]
        evals = [(xgb.DMatrix(X[rv], yv, weight=s[rv] if ratio else None), "val")]
    bst = xgb.train(p, dtr, n, evals=evals, evals_result=ev, verbose_eval=False)
    return (bst, np.array(ev["val"]["mae"])) if va is not None else bst


def predict_xgb(bst, X, S, ratio, d0, d1, n=None):
    r = row_index(d0, d1)
    p = bst.predict(xgb.DMatrix(X[r]), iteration_range=(0, n) if n else (0, 0))
    if ratio:
        p = p * S.reshape(-1)[r]
    return p.reshape(d1 - d0 + 1, 24, Z)


def fit_lgb_fixed(X, names, Y, S, ratio, tr, params, n):
    y = Y.reshape(-1)
    s = S.reshape(-1)
    rt = row_index(*tr)
    ytr = y[rt] / s[rt] if ratio else y[rt]
    d = lgb.Dataset(
        X[rt],
        ytr,
        weight=s[rt] if ratio else None,
        categorical_feature=[names.index("zone")],
        feature_name=names,
    )
    return lgb.train(params, d, n)
