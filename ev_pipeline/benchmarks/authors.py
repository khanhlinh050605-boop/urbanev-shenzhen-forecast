"""Run the upstream statistical benchmarks in a subprocess."""
import os
import sys
import time
from pathlib import Path

from .upstream_sources import STATISTICAL_RUN, STATISTICAL_SOURCES
from ..config import ARIMA_TIME_LIMIT_SECONDS
from ..utils.io import save_json


def authors_benchmarks(data_path, out, reuse):
    import subprocess
    import tempfile

    out = Path(out)
    if reuse:
        if not (out / "authors" / "lo_ar" / "truth_fold1_h3.npy").exists():
            raise FileNotFoundError("Missing statistical benchmark predictions")
        return
    with tempfile.TemporaryDirectory(prefix="urbanev_statistics_") as folder:
        worker = Path(folder) / "statistics.py"
        worker.write_text(
            "STATISTICAL_SOURCES = " + repr(STATISTICAL_SOURCES) + "\n" + STATISTICAL_RUN,
            encoding="utf-8",
        )
        env = dict(
            os.environ,
            STAT_ROOT=str(out.parent.resolve()),
            URBANEV_OUT=str(out.resolve()),
            URBANEV_DATA=str(Path(data_path).resolve()),
        )
        subprocess.run([sys.executable, str(worker), "--models", "lo", "ar"], env=env, check=True)
        process = subprocess.Popen(
            [sys.executable, str(worker), "--models", "arima", "--horizons", "3"], env=env
        )
        started = time.monotonic()
        try:
            status = process.wait(timeout=ARIMA_TIME_LIMIT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()
            status = "time_limit_reached"
        save_json(
            out / "authors" / "arima_budget.json",
            dict(status=status, elapsed_seconds=time.monotonic() - started),
        )
