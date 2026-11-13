"""Embedded upstream benchmark source code (kept as strings, hashed in provenance)."""


STATISTICAL_SOURCES = {
    "baselines.py": r"""class Lo:

    def __init__(self, args):
        self.args = args
        self.pred_len = args.pred_len

    def predict(self, train_valid_occ, test_occ):
        time_len, node = test_occ.shape
        preds = np.zeros((time_len, node))
        for j in range(node):
            for i in range(time_len):
                if i < self.pred_len:
                    preds[i, j] = train_valid_occ[-self.pred_len + i, j]
                else:
                    preds[i, j] = test_occ[i - self.pred_len, j]
        return preds

class Ar:

    def __init__(self, pred_len, args, lags=1):
        self.args = args
        self.pred_len = pred_len
        self.lags = lags

    def predict(self, train_valid_occ, test_occ):
        time_len, node = test_occ.shape
        train_valid_occ = train_valid_occ[:-self.pred_len, :]
        preds = np.zeros((time_len, node))
        for j in range(node):
            series = list(train_valid_occ[:, j])
            for i in range(time_len):
                try:
                    model = AutoReg(series, lags=self.lags).fit()
                    pred = model.predict(start=len(series), end=len(series))
                    preds[i, j] = pred[0]
                    series.append(pred[0])
                except Exception as e:
                    print(f'[AutoReg] Failed at node {j}, step {i}: {e}')
                    preds[i, j] = np.nan
        return preds

class Arima:

    def __init__(self, pred_len, args, p=1, d=1, q=1):
        self.pred_len = pred_len
        self.args = args
        self.p = p
        self.d = d
        self.q = q

    def predict(self, train_valid_occ, test_occ):
        time_len, node = test_occ.shape
        train_valid_occ = train_valid_occ[:-self.pred_len, :]
        preds = np.zeros((time_len, node))
        warnings.filterwarnings('ignore')
        for j in range(node):
            fit_series = train_valid_occ[:, j]
            try:
                model = ARIMA(fit_series, order=(self.p, self.d, self.q))
                model_fitted = model.fit()
                pred = model_fitted.forecast(steps=time_len)
                preds[:, j] = pred
            except Exception as e:
                print(f'ARIMA failed for node {j}: {e}')
                preds[:, j] = np.nan
        return preds""",
    "utils.py": r"""def read_data(args):
    inf = pd.read_csv('../data/inf.csv', header=0, index_col=None)
    occ = pd.read_csv('../data/occupancy.csv', header=0, index_col=0)
    duration = pd.read_csv('../data/duration.csv', header=0, index_col=0)
    volume = pd.read_csv('../data/volume.csv', header=0, index_col=0)
    e_price = pd.read_csv('../data/e_price.csv', index_col=0, header=0).values
    s_price = pd.read_csv('../data/s_price.csv', index_col=0, header=0).values
    adj = pd.read_csv('../data/adj.csv', header=0, index_col=None)
    adj.index = adj.columns
    time = pd.to_datetime(occ.index)
    feat = occ
    if args.feat == 'duration':
        feat = duration
    elif args.feat == 'volume':
        feat = volume
    charge_count_dict = dict(zip(inf['TAZID'].astype(str), inf['charge_count']))
    for col in occ.columns:
        charge_count = charge_count_dict[col]
        occ[col] = occ[col] / charge_count
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    e_price = price_scaler.fit_transform(e_price)
    s_price = price_scaler.fit_transform(s_price)
    weather = pd.read_csv('../data/weather_central.csv', header=0, index_col='time')
    extra_feat = 'None'
    if args.add_feat != 'None':
        extra_feat = np.zeros([occ.shape[0], occ.shape[1], 1])
        add_feat_list = args.add_feat.split('+')
        for add_feat in add_feat_list:
            if add_feat == 'all':
                extra_feat = np.concatenate([extra_feat, e_price[:, :, np.newaxis]], axis=2)
                extra_feat = np.concatenate([extra_feat, s_price[:, :, np.newaxis]], axis=2)
                extra_feat = np.concatenate([extra_feat, np.repeat(weather.values[:, np.newaxis, :], occ.shape[1], axis=1)], axis=2)
            elif add_feat == 'e':
                extra_feat = np.concatenate([extra_feat, e_price[:, :, np.newaxis]], axis=2)
            elif add_feat == 's':
                extra_feat = np.concatenate([extra_feat, s_price[:, :, np.newaxis]], axis=2)
            else:
                extra_feat = np.concatenate([extra_feat, np.repeat(weather[add_feat].values[:, np.newaxis, np.newaxis], occ.shape[1], axis=1)], axis=2)
        extra_feat = extra_feat[:, :, 1:]
    return (np.array(feat), np.array(adj), extra_feat, time)

def metrics(test_pre, test_real, args):
    eps = 0.02
    MAPE_test_real = test_real.copy()
    MAPE_test_pre = test_pre.copy()
    MAPE_test_real[np.where(MAPE_test_real <= eps)] = np.abs(MAPE_test_real[np.where(MAPE_test_real <= eps)]) + eps
    MAPE_test_pre[np.where(MAPE_test_real <= eps)] = np.abs(MAPE_test_pre[np.where(MAPE_test_real <= eps)]) + eps
    MAPE = mean_absolute_percentage_error(MAPE_test_real, MAPE_test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    RAE = np.sum(abs(MAPE_test_pre - MAPE_test_real)) / np.sum(abs(np.mean(MAPE_test_real) - MAPE_test_real))
    print('MAPE: {}'.format(MAPE))
    print('MAE:{}'.format(MAE))
    print('MSE:{}'.format(MSE))
    print('RMSE:{}'.format(RMSE))
    print('RAE:{}'.format(RAE))
    output_list = [MSE, RMSE, MAPE, RAE, MAE]
    return output_list

def split_cv(args, time, feat, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, extra_feat='None'):
    assert len(time) == len(feat)
    fold = args.fold
    month_list = list(time.month.unique())
    assert args.total_fold == len(month_list)
    fold_time = time.month.isin(month_list[0:fold]).sum()
    train_end = int(fold_time * train_ratio)
    valid_start = train_end
    valid_end = int(valid_start + fold_time * valid_ratio)
    test_start = valid_end
    test_end = int(fold_time)
    train_feat = feat[:train_end]
    valid_feat = feat[valid_start:valid_end]
    test_feat = feat[test_start:test_end]
    scaler = 'None'
    if args.pred_type == 'region':
        if args.feat != 'occ':
            scaler = StandardScaler()
            train_feat = scaler.fit_transform(train_feat)
            valid_feat = scaler.transform(valid_feat)
            test_feat = scaler.transform(test_feat)
    else:
        node_idx = int(args.pred_type)
        if args.feat != 'occ':
            scaler = StandardScaler()
            train_feat = scaler.fit_transform(train_feat[:, node_idx].reshape(-1, 1))
            valid_feat = scaler.transform(valid_feat[:, node_idx].reshape(-1, 1))
            test_feat = scaler.transform(test_feat[:, node_idx].reshape(-1, 1))
        else:
            train_feat = train_feat[:, node_idx].reshape(-1, 1)
            valid_feat = valid_feat[:, node_idx].reshape(-1, 1)
            test_feat = test_feat[:, node_idx].reshape(-1, 1)
    train_extra_feat, valid_extra_feat, test_extra_feat = ('None', 'None', 'None')
    if extra_feat != 'None':
        train_extra_feat = extra_feat[:train_end]
        valid_extra_feat = extra_feat[valid_start:valid_end]
        test_extra_feat = extra_feat[test_start:test_end]
    return (train_feat, valid_feat, test_feat, train_extra_feat, valid_extra_feat, test_extra_feat, scaler)""",
}


STATISTICAL_RUN = r"""from pathlib import Path
from types import SimpleNamespace
import argparse, ast, hashlib, json, os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
ROOT = Path(os.environ['STAT_ROOT'])
OUT = Path(os.environ['URBANEV_OUT'])
write_json = lambda p, v: Path(p).write_text(json.dumps(v, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', choices=['lo', 'ar', 'arima'], default=['lo', 'ar'])
parser.add_argument('--horizons', nargs='+', type=int, default=[3, 6, 9, 12])
parser.add_argument('--exact-ar', action='store_true')
opts = parser.parse_args()
models = set(opts.models)
horizons = sorted(opts.horizons)
repo = Path(os.environ['URBANEV_DATA']).parent
source = repo / 'code'
od = OUT / 'authors' / '_'.join(opts.models)
od.mkdir(parents=True, exist_ok=True)
ns = dict(np=np, pd=pd, os=os, warnings=warnings, AutoReg=AutoReg, ARIMA=ARIMA, mean_squared_error=mean_squared_error, mean_absolute_error=mean_absolute_error, mean_absolute_percentage_error=mean_absolute_percentage_error, r2_score=r2_score, MinMaxScaler=MinMaxScaler, StandardScaler=StandardScaler)
fit_diagnostics = []
fit_context = {}

def traced_arima(*args, **kwargs):
    model = ARIMA(*args, **kwargs)
    original_fit = model.fit

    def fit(*fitargs, **fitkwargs):
        started = time.monotonic()
        result = original_fit(*fitargs, **fitkwargs)
        info = getattr(result, 'mle_retvals', {})
        fit_diagnostics.append(dict(**fit_context, converged=info.get('converged'), iterations=info.get('iterations'), warnflag=info.get('warnflag'), elapsed_seconds=time.monotonic() - started))
        return result
    model.fit = fit
    return model
hashes = {}
for file, source_text in STATISTICAL_SOURCES.items():
    exec(source_text.replace('../data', os.environ['URBANEV_DATA'].replace('\\', '/')), ns)
ns['ARIMA'] = traced_arima
args = SimpleNamespace(seq_len=12, pred_len=3, total_fold=6, fold=1, feat='occ', pred_type='region', add_feat='None')
feat, adj, extra, timeindex = ns['read_data'](args)
inf = pd.read_csv(Path(os.environ['URBANEV_DATA']) / 'inf.csv')
write_json(od / 'provenance.json', dict(source_hashes=hashes, commit='44f2aa0c8d89f192bce00bafb0def74a21b39c68', data_fingerprints='results/data_provenance.json', occupancy_normalization='Unmodified upstream dict(zip(TAZID, charge_count)); duplicate zone keys keep the last station row, rather than summing zone piles.', input_hours=12, horizons=horizons, models=opts.models, folds=6, split=[0.8, 0.1, 0.1], features='occupancy only', acceleration='Reuse identical fit data across horizons; AR fixed OLS recursion verified against upstream refits', exact_ar='--exact-ar' in sys.argv))

def ar_fast(train, n):
    fit = AutoReg(train, lags=12).fit()
    beta = np.asarray(fit.params)
    series = list(train)
    out = []
    for _ in range(n):
        p = float(beta[0] + np.dot(beta[1:], np.asarray(series[-12:][::-1])))
        out.append(p)
        series.append(p)
    return np.asarray(out)
checks = []
rows = []
for fold in range(1, 7):
    args.fold = fold
    tr, va, te, *_, scaler = ns['split_cv'](args, timeindex, feat, 0.8, 0.1, 0.1, extra)
    fit = np.vstack([tr, va, te[:12]])
    first = min(horizons)
    nmax = len(te) - 12 - first
    ar = np.full((nmax, feat.shape[1]), np.nan)
    arima = np.full_like(ar, np.nan)
    for zone in sorted(set([0, 1, int(np.argmax(feat.mean(0))), int(np.argmin(feat.mean(0)))])) if 'ar' in models else []:
        exact = ns['Ar'](3, args, lags=12).predict(np.vstack([fit[:, [zone]], np.zeros((3, 1))]), np.zeros((nmax, 1)))[:, 0]
        fast = ar_fast(fit[:, zone], len(exact))
        err = float(np.max(np.abs(exact - fast)))
        checks.append(dict(fold=fold, zone_index=zone, forecast_hours=len(exact), max_abs_difference=err, passed=bool(np.allclose(exact, fast, rtol=1e-08, atol=1e-09))))
    if 'ar' in models:
        write_json(od / 'ar_equivalence_checks.json', checks)
    if not all((x['passed'] for x in checks)):
        raise RuntimeError('AR acceleration failed exact-source check')
    for z in range(feat.shape[1]):
        if 'ar' in models:
            if opts.exact_ar:
                ar[:, z] = ns['Ar'](first, args, lags=12).predict(np.vstack([fit[:, [z]], np.zeros((first, 1))]), np.zeros((nmax, 1)))[:, 0]
            else:
                ar[:, z] = ar_fast(fit[:, z], nmax)
        if 'arima' in models:
            fit_context.update(fold=fold, zone_index=z)
            arima[:, z] = ns['Arima'](first, args, p=12).predict(np.vstack([fit[:, [z]], np.zeros((first, 1))]), np.zeros((nmax, 1)))[:, 0]
        if z % 25 == 0:
            print('authors fold', fold, 'zone', z, 'of', feat.shape[1], flush=True)
            write_json(od / 'progress.json', dict(fold=fold, zones_completed=z + 1, models=opts.models))
            if 'arima' in models:
                np.save(od / f'arima_partial_fold{fold}.npy', arima.astype(np.float32))
            if fit_diagnostics:
                pd.DataFrame(fit_diagnostics).to_csv(od / 'fit_diagnostics.csv', index=False)
    for horizon in horizons:
        args.pred_len = horizon
        actual = te[12 + horizon:]
        tv = np.vstack([tr, va, te[:12 + horizon]])
        predictions = {}
        if 'lo' in models:
            predictions['lo'] = ns['Lo'](args).predict(tv, actual)
        if 'ar' in models:
            predictions['ar'] = ar[:len(actual)]
        if 'arima' in models:
            predictions['arima'] = arima[:len(actual)]
        for name, pred in predictions.items():
            np.save(od / f'{name}_fold{fold}_h{horizon}.npy', pred.astype(np.float32))
            if np.isfinite(pred).all():
                vals = ns['metrics'](pred.copy(), actual.copy(), args)
                row = dict(zip(['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE'], vals))
                row['status'] = 'complete'
            else:
                row = dict(status='failed_nonfinite', nonfinite_cells=int((~np.isfinite(pred)).sum()))
            rows.append(dict(model=name, fold=fold, horizon_hours=horizon, **row))
        np.save(od / f'truth_fold{fold}_h{horizon}.npy', actual.astype(np.float32))
    pd.DataFrame(rows).to_csv(od / 'results.csv', index=False)
    if fit_diagnostics:
        pd.DataFrame(fit_diagnostics).to_csv(od / 'fit_diagnostics.csv', index=False)
    print('authors fold', fold, 'complete', flush=True)
frame = pd.DataFrame(rows).reindex(columns=['model', 'fold', 'horizon_hours', 'status', 'MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'nonfinite_cells'])
frame.groupby(['model', 'horizon_hours'])[['RMSE', 'MAPE', 'RAE', 'MAE']].mean().reset_index().to_csv(od / 'summary.csv', index=False)
print('DONE authors statistical reproduction', flush=True)"""
