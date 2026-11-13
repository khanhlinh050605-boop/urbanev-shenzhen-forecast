"""Design matrix for the ridge model."""
import numpy as np

from ..config import Z
from ..features.matrix import row_index


def ridge_inputs(X, names, cal, a, b):
    excluded = {"zone", "cc", "nst", "area", "perim", "hour", "dow", "log_s"}
    selected = [
        i for i, n in enumerate(names) if n not in excluded and not n.startswith(("nb_", "city_"))
    ]
    hours = np.eye(24, dtype=np.float32)[np.tile(np.repeat(np.arange(24), Z), b - a + 1)]
    days = np.eye(7, dtype=np.float32)[np.repeat(cal.dow.values[a : b + 1], 24 * Z)]
    return np.hstack([X[row_index(a, b)][:, selected], hours, days])
