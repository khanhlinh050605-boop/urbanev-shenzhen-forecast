"""Stage-by-stage run: data checks, fitting, evaluation, decisions, figures."""
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.tsa.ar_model
import statsmodels.tsa.arima.model
import xgboost as xgb

from .analysis.calibration import calibrate_quantiles
from .analysis.decisions import decision_analysis
from .analysis.investigation import investigate
from .analysis.report_numbers import report_numbers
from .analysis.leakage import leakage_checks, original_floor_controls
from .analysis.prediction_metrics import evaluate_predictions, prediction_inventory
from .analysis.reference import reference_comparison
from .analysis.reproduction import refresh_statistical_tables, reproduction_details
from .benchmarks.authors import authors_benchmarks
from .benchmarks.normalization import normalization_benchmark
from .config import Settings
from .data.loader import load_data
from .data.provenance import descriptive_evidence, save_data_provenance
from .data.validation import clean_data
from .models.extended import extended_fits
from .models.training import fit_comparison_lightgbm, fit_predictions
from .reporting.manifest import prediction_hashes, run_manifest, source_fingerprint
from .utils.io import record_progress, save_json
from .viz.merged import merged
from .viz.runner import visualize


STAGES = ("setup", "clean_data", "models", "evaluate", "decision_analysis", "investigate", "visualize", "merged_figures", "finish", "report_numbers")


@dataclass
class RunState:
    settings: Settings
    results: Path
    started: float
    manifest: dict
    data: dict = None
    figures: list = None


def stage_setup(settings):
    results = Path(settings.output_path) / "results"
    state = RunState(
        settings=settings,
        results=results,
        started=time.monotonic(),
        manifest=run_manifest(settings.data_path, results, settings.seed),
    )
    if settings.reuse_saved_predictions:
        previous = json.loads((results / "prediction_manifest.json").read_text())
        if previous != json.loads(json.dumps(state.manifest)):
            raise ValueError("Saved prediction configuration or input mismatch")
    results.mkdir(parents=True, exist_ok=True)
    return state


def stage_clean_data(state):
    s, results = state.settings, state.results
    state.data = clean_data(load_data(s.data_path), s.data_path, results)
    save_data_provenance(s.data_path, results)
    descriptive_evidence(state.data, results)
    save_json(
        results / "execution_provenance.json",
        dict(
            python=platform.python_version(),
            packages={
                m.__name__: m.__version__ for m in [np, pd, scipy, lgb, xgb, sklearn, statsmodels]
            },
            seed=s.seed,
            source_sha256=source_fingerprint(),
            reuse_predictions=bool(s.reuse_saved_predictions),
            test_label="final test",
        ),
    )
    if not s.reuse_saved_predictions:
        save_json(
            results / "fit_provenance.json",
            json.loads((results / "execution_provenance.json").read_text()),
        )
    leakage_checks(state.data, results)
    original_floor_controls(state.data, results)


def stage_models(state):
    s, results, D = state.settings, state.results, state.data
    if s.reuse_saved_predictions:
        missing = [
            str(p.relative_to(results)) for p in prediction_inventory(results) if not p.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing saved predictions: " + ", ".join(missing))
    else:
        fit_predictions(D, results, s.seed)
        fit_comparison_lightgbm(D, results, s.seed)
    extended_fits(D, results, s.seed, s.reuse_saved_predictions)


def stage_evaluate(state):
    s, results, D = state.settings, state.results, state.data
    record_progress(results, phase="evaluation")
    evaluate_predictions(D, results, s.seed)
    calibrate_quantiles(D, results, s.seed)


def stage_decision_analysis(state):
    s, results, D = state.settings, state.results, state.data
    decision_analysis(D, results, s.seed)
    reference_comparison(D, results)
    reproduction_details(D, results)
    authors_benchmarks(s.data_path, results, s.reuse_saved_predictions)
    if not s.reuse_saved_predictions:
        normalization_benchmark(s.data_path, results)
    elif not (
        results / "published_baselines" / "zone_total_diagnostic" / "truth_fold1_h3.npy"
    ).exists():
        raise FileNotFoundError("Missing normalised benchmark predictions")
    refresh_statistical_tables(results)
    record_progress(
        results,
        phase="complete",
        elapsed_seconds=time.monotonic() - state.started,
        reuse_predictions=bool(s.reuse_saved_predictions),
        test_label="final test",
    )
    state.manifest["prediction_sha256"] = prediction_hashes(results)
    save_json(results / "prediction_manifest.json", state.manifest)


def stage_investigate(state):
    investigate(state.data, state.results, state.settings.seed)


def stage_merged_figures(state):
    merged(state.settings.output_path, state.results, state.data)


def stage_report_numbers(state):
    report_numbers(state.results)


def stage_visualize(state):
    s = state.settings
    state.data["data_dir"] = Path(s.data_path)
    figures = visualize(state.data, Path(s.output_path), state.results)
    state.figures = []
    # One ``next`` per figure, as in the original cells: the generator is never run to
    # completion, so the trailing axis/text/protocol tables inside ``visualize`` are not written.
    for _ in range(N_FIGURES):
        state.figures.append(next(figures))


def stage_finish(state):
    if len(state.figures) != N_FIGURES:
        raise RuntimeError("Expected %d figures" % N_FIGURES)
    save_json(
        state.results / "execution_summary.json",
        dict(
            elapsed_seconds=time.monotonic() - state.started,
            seed=state.settings.seed,
            reuse_predictions=bool(state.settings.reuse_saved_predictions),
            figures=len(state.figures),
            test_label="final test",
        ),
    )


N_FIGURES = 15


def run(settings):
    state = stage_setup(settings)
    stage_clean_data(state)
    stage_models(state)
    stage_evaluate(state)
    stage_decision_analysis(state)
    stage_investigate(state)
    stage_visualize(state)
    stage_merged_figures(state)
    stage_finish(state)
    stage_report_numbers(state)
    return state
