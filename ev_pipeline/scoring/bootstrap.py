"""Block bootstrap helpers."""
import numpy as np


def bootstrap_indices(n, block=7, draws=5000, seed=42):
    rng = np.random.default_rng(seed)
    if block == 1:
        return rng.integers(0, n, (draws, n))
    starts = rng.integers(0, n - block + 1, (draws, int(np.ceil(n / block))))
    return (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :n]


def boot_ci(d, B=5000, block=1, seed=42):
    values = np.asarray(d)[bootstrap_indices(len(d), block, B, seed)].mean(1)
    return (
        float(np.mean(d)),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float((values < 0).mean()),
    )
