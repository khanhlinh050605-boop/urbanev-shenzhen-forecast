"""Zone clustering evidence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ND, Z
from ..utils.io import save_json, write_table


def cluster_evidence(D, out, seed, reuse):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    out = Path(out)
    choices = []
    train = D["V"].reshape(ND, 24, Z)[:30]
    profiles = train.mean(0).T
    profiles /= profiles.mean(1, keepdims=True)
    cache = out / "cluster_fit.json"
    if reuse:
        saved = json.loads(cache.read_text())
        if saved["seed"] != seed:
            raise ValueError("Clustering seed mismatch")
        choices = saved["choices"]
        labels = np.asarray(saved["labels"], dtype=int)
    else:
        for k in range(2, 7):
            labels = [
                KMeans(k, n_init=20, random_state=x).fit_predict(profiles)
                for x in (seed, 1, 2, 3, 4)
            ]
            rng = np.random.default_rng(seed)
            scores = []
            for _ in range(10):
                ix = (rng.integers(0, 24, 5)[:, None] + np.arange(7)).reshape(-1)[:30]
                pp = train[ix].mean(0).T
                pp /= pp.mean(1, keepdims=True)
                scores.append(
                    adjusted_rand_score(
                        labels[0], KMeans(k, n_init=20, random_state=seed).fit_predict(pp)
                    )
                )
            choices.append(
                dict(
                    k=k,
                    silhouette=silhouette_score(profiles, labels[0]),
                    seed_stability_ARI=np.mean(
                        [adjusted_rand_score(labels[0], x) for x in labels[1:]]
                    ),
                    block_bootstrap_stability_ARI=np.mean(scores),
                )
            )
        labels = KMeans(2, n_init=20, random_state=seed).fit_predict(profiles)
        save_json(cache, dict(seed=seed, choices=choices, labels=labels.tolist()))
    write_table(out, "cluster_selection", choices)
    write_table(
        out,
        "zone_clusters",
        pd.DataFrame(
            dict(
                zone=D["zones"],
                cluster=labels,
                training_mean_kwh=train.mean((0, 1)),
                training_volume_share=train.sum((0, 1)) / train.sum(),
            )
        ),
    )
    write_table(
        out, "cluster_profiles", pd.DataFrame(profiles).assign(zone=D["zones"], cluster=labels)
    )
    write_table(
        out,
        "cluster_summary",
        [
            dict(
                cluster=int(k),
                zones=int((labels == k).sum()),
                peak_hour=int(profiles[labels == k].mean(0).argmax()),
                training_volume_share=float(train[:, :, labels == k].sum() / train.sum()),
            )
            for k in np.unique(labels)
        ],
    )
