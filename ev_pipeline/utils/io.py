"""Small helpers for writing arrays, tables, JSON and progress files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)
        ),
        encoding="utf-8",
    )


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array, dtype=np.float32))


def write_table(out, name, rows):
    path = Path(out) / (name + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def record_progress(out, **values):
    save_json(Path(out) / "progress.json", values)
