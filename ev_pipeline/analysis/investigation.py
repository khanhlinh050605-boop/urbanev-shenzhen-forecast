"""Post-hoc diagnostics (no setting is selected here): why do some zones lose to yesterday,
and what does each step of the model ladder add. Reads saved predictions and daily losses only."""
import numpy as np
import pandas as pd
from scipy import stats

from ..config import TEST

TARGETS = ("volume", "occ")
LADDER = (
    ("lgb_base", "lgb_nbr", "Add neighbour / city features"),
    ("lgb_nbr", "lgb_nbr_ratio", "Ratio-scaled target"),
    ("lgb_nbr_ratio", "lgb_training_floor", "Training-only floor"),
    ("lgb_training_floor", "blend_training_floor", "Blend with XGBoost"),
    ("lgb_base", "blend_training_floor", "Total: plain LightGBM to blend"),
)


def _blocks(n, seed=42):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, max(n - 6, 1), (5000, int(np.ceil(n / 7))))
    return (s[:, :, None] + np.arange(7)).reshape(5000, -1)[:, :n].clip(max=n - 1)


def _auc(x, y):
    u = stats.mannwhitneyu(x[y], x[~y])
    return u.statistic / (y.sum() * (~y).sum()), u.pvalue


def _arrays(data):
    V = np.asarray(data["V"], float).reshape(-1, 24, len(data["zones"]))
    cc = np.asarray(data["st"]["cc"], float)
    O = np.asarray(data["O"], float).reshape(V.shape) / cc
    return {"volume": V, "occ": O}


def zone_failure(data, results, seed=42):
    arrays = _arrays(data)
    zones = [str(z) for z in data["zones"]]
    t0 = TEST[0]
    rng = np.random.default_rng(seed)
    diag, tests, conc, traj = [], [], [], []
    for target in TARGETS:
        X = arrays[target]
        A, tr, last = X[t0:], X[:t0], X[t0 - 30 : t0]
        Pb = np.load(results / "extra" / "floor" / target / "blend_test.npy").astype(float)
        Ps = np.load(results / "test" / target / "snaive24.npy").astype(float)
        mb, ms = np.abs(Pb - A).mean((0, 1)), np.abs(Ps - A).mean((0, 1))
        skill = 1 - mb / ms
        lose = skill < 0
        m = tr.mean((0, 1)).clip(1e-9)
        d = pd.DataFrame(
            dict(
                target=target, zone=zones, skill=skill, loses=lose, excess=mb - ms,
                train_mean=m, train_cv=tr.mean(1).std(0) / m,
                zero_share=(tr <= 1e-9).mean((0, 1)),
                last30_vs_train=last.mean((0, 1)) / m,
                test_vs_last30=A.mean((0, 1)) / last.mean((0, 1)).clip(1e-9),
                festival_ratio=A[1:8].mean((0, 1)) / np.delete(A, range(1, 8), 0).mean((0, 1)).clip(1e-9),
                max_daily_jump_final=np.abs(np.diff(A.mean(1), axis=0)).max(0) / m,
                max_daily_jump_train=np.abs(np.diff(tr.mean(1), axis=0)).max(0) / m,
            )
        )
        diag.append(d)
        for c in ["train_mean", "train_cv", "zero_share", "last30_vs_train", "test_vs_last30",
                  "festival_ratio", "max_daily_jump_final", "max_daily_jump_train"]:
            x = d[c].to_numpy()
            a, p = _auc(x, lose)
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(x), len(x))
                if lose[i].sum() >= 2 and (~lose[i]).sum() >= 2:
                    bs.append(_auc(x[i], lose[i])[0])
            tests.append(dict(target=target, covariate=c, losing_zones=int(lose.sum()),
                              median_losing=np.median(x[lose]), median_other=np.median(x[~lose]),
                              auc=a, auc_lower=np.quantile(bs, .025), auc_upper=np.quantile(bs, .975),
                              mann_whitney_p=p))
        L = d[lose].sort_values("excess", ascending=False)
        for k in (1, 5, 10, len(L)):
            conc.append(dict(target=target, top_losing_zones=k,
                             excess_share=float(L.excess.head(k).sum() / L.excess.sum()),
                             excess_mae=float(L.excess.head(k).sum())))
        base = last.mean((0, 1))
        dates = pd.date_range("2022-09-01", periods=len(X))[t0:]
        for name, mask in (("losing", lose), ("other", ~lose)):
            den = base[mask].sum() * 24
            for key, P in (("observed", A), ("blend", Pb), ("yesterday", Ps)):
                for day, v in zip(dates, P[:, :, mask].sum((1, 2)) / den):
                    traj.append(dict(target=target, group=name, series=key, date=day, relative_level=v))
    pd.concat(diag).to_csv(results / "zone_failure_diagnostics.csv", index=False)
    pd.DataFrame(tests).to_csv(results / "zone_failure_tests.csv", index=False)
    pd.DataFrame(conc).to_csv(results / "zone_failure_concentration.csv", index=False)
    pd.DataFrame(traj).to_csv(results / "zone_failure_trajectory.csv", index=False)


def ladder_ablation(results):
    d = pd.read_csv(results / "daily_losses.csv")
    rows = []
    for t in TARGETS:
        for s in ("val", "test"):
            for a, b, lab in LADDER:
                sel = d[(d.target == t) & (d.split == s)]
                x = sel[sel.model == a].set_index("day").zone_MAE
                y = sel[sel.model == b].set_index("day").zone_MAE
                x, y = x.align(y, join="inner")
                diff = (y - x).to_numpy()
                bs = diff[_blocks(len(diff))].mean(1)
                base = x.mean()
                rows.append(dict(target=t, split=s, step=lab, from_model=a, to_model=b, days=len(diff),
                                 difference=diff.mean(), lower=np.quantile(bs, .025), upper=np.quantile(bs, .975),
                                 percent=100 * diff.mean() / base,
                                 percent_lower=100 * np.quantile(bs, .025) / base,
                                 percent_upper=100 * np.quantile(bs, .975) / base))
    pd.DataFrame(rows).to_csv(results / "ablation_steps.csv", index=False)


def investigate(data, results, seed=42):
    zone_failure(data, results, seed)
    ladder_ablation(results)
