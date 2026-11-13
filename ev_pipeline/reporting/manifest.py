"""Run manifest, fit signature and prediction hashes."""
import ast
import hashlib
from pathlib import Path

from ..config import FOLDS, HOLIDAYS, LAGS, LOCKED, MAKEUP, QUANTILES, START, TEST
from ..utils.io import file_sha


def prediction_hashes(results):
    results = Path(results)
    metadata = {
        "cluster_fit.json",
        "fit_provenance.json",
        "floor_constants.json",
        "frozen_configuration.json",
        "comparison_configuration.json",
        "selection.json",
        "fit_diagnostics.csv",
        "ar_equivalence_checks.json",
        "diagnostic_ar_equivalence_checks.json",
        "arima_budget.json",
        "provenance.json",
        "diagnostic_provenance.json",
    }
    files = [
        p for p in results.rglob("*") if p.is_file() and (p.suffix == ".npy" or p.name in metadata)
    ]
    return {str(p.relative_to(results)): file_sha(p) for p in sorted(files)}


def run_manifest(data_path, results, seed):
    return dict(
        seed=seed,
        models=LOCKED,
        quantiles=QUANTILES,
        input_sha256={p.name: file_sha(p) for p in Path(data_path).glob("*.csv")},
        fit_signature=fit_signature(),
        prediction_sha256=prediction_hashes(results),
        protocol=dict(
            start=str(START),
            lags=LAGS,
            folds=FOLDS,
            final_window=TEST,
            holidays=HOLIDAYS,
            makeup=MAKEUP,
        ),
    )

_SIGNATURE_FUNCTIONS = ['authors_benchmarks', 'baseline_preds', 'build_features', 'build_split', 'calendar', 'cluster_evidence', 'extended_fits', 'fit_comparison_lightgbm', 'fit_lgb_fixed', 'fit_predictions', 'fit_xgb', 'lgb_params', 'load_data', 'normalization_benchmark', 'predict_lgb', 'predict_xgb', 'ridge_inputs', 'row_index', 'sh', 'split_specs', 'to_matrix']
_SIGNATURE_CONSTANTS = ['FOLDS', 'HOLIDAYS', 'LAGS', 'MAKEUP', 'MIN_DAY', 'ND', 'NH', 'START', 'STATISTICAL_RUN', 'STATISTICAL_SOURCES', 'TEST', 'Z']
# Definition order of the signed nodes in the original single-file script; keeping it
# makes the hash match signatures recorded by that version.
_SIGNATURE_ORDER = ['ND', 'START', 'LAGS', 'load_data', 'calendar', 'sh', 'build_features', 'to_matrix', 'row_index', 'build_split', 'MIN_DAY', 'FOLDS', 'TEST', 'HOLIDAYS', 'MAKEUP', 'split_specs', 'baseline_preds', 'lgb_params', 'predict_lgb', 'fit_xgb', 'predict_xgb', 'fit_lgb_fixed', 'fit_comparison_lightgbm', 'ridge_inputs', 'fit_predictions', 'cluster_evidence', 'extended_fits', 'STATISTICAL_SOURCES', 'STATISTICAL_RUN', 'authors_benchmarks', 'normalization_benchmark']


def _signature_key(node):
    if isinstance(node, ast.FunctionDef) and node.name in _SIGNATURE_FUNCTIONS:
        return node.name
    if isinstance(node, ast.Assign):
        hit = sorted(
            {x.id for t in node.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
            & set(_SIGNATURE_CONSTANTS)
        )
        if hit:
            return hit[0]
    return None


def _package_files():
    root = Path(__file__).resolve().parents[1]
    return sorted(
        p for p in root.rglob("*.py")
        if not {".venv", "__pycache__"} & set(p.relative_to(root).parts)
    )


def source_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in _package_files():
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fit_signature():
    found = {}
    for path in _package_files():
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            key = _signature_key(node)
            if key is not None:
                found[key] = node
    nodes = [found[k] for k in _SIGNATURE_ORDER if k in found]
    return hashlib.sha256(
        ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()
    ).hexdigest()
