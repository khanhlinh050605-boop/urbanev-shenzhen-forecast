"""Commitment-decision cost analysis."""
from pathlib import Path

import numpy as np

from ..config import FOLDS, QUANTILES, RATIOS, Z
from ..scoring.bootstrap import bootstrap_indices
from ..scoring.splits import evaluation
from ..utils.io import save_json, write_table


def asymmetric_cost(A, Q, ratio):
    return ratio * np.maximum(A - Q, 0) + np.maximum(Q - A, 0)


def fit_safety_factor(A, P, ratio):
    a = np.asarray(A, dtype=np.float64).ravel()
    p = np.asarray(P, dtype=np.float64).ravel()
    nz = p > 0
    if not nz.any():
        return 1.0
    values = a[nz] / p[nz]
    weights = p[nz]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    j = min(
        np.searchsorted(cumulative, ratio / (1 + ratio) * cumulative[-1], side="left"),
        len(values) - 1,
    )
    k = max(0.0, float(values[j]))
    left = np.searchsorted(values, k, side="left")
    right = np.searchsorted(values, k, side="right")
    before = float(cumulative[left - 1]) if left else 0.0
    through = float(cumulative[right - 1])
    total = float(cumulative[-1])
    tolerance = 1e-10 * max(1.0, (1 + ratio) * total)
    if (1 + ratio) * before - ratio * total > tolerance or (
        1 + ratio
    ) * through - ratio * total < -tolerance:
        raise ArithmeticError("Safety-factor optimum failed")
    return k


def cost_summary(day, baseline, seed):
    row = dict(
        evaluation_days=len(day),
        mean_daily_cost_units=float(day.mean()),
        relative_cost_difference=float(day.sum() / baseline.sum() - 1),
        daily_cost_difference=float((day - baseline).mean()),
    )
    for block, name in [(1, "day"), (7, "block7")]:
        ix = bootstrap_indices(len(day), block, seed=seed)
        relative = day[ix].sum(1) / baseline[ix].sum(1) - 1
        difference = (day - baseline)[ix].mean(1)
        row.update(
            {
                f"relative_{name}_lower": float(np.quantile(relative, 0.025)),
                f"relative_{name}_upper": float(np.quantile(relative, 0.975)),
                f"{name}_lower": float(np.quantile(difference, 0.025)),
                f"{name}_upper": float(np.quantile(difference, 0.975)),
            }
        )
    return row


def decision_analysis(D, out, seed):
    evs = {split: evaluation(D, "volume", split) for split in ("val", "test")}
    alphas = [a for a, _ in QUANTILES["volume"]]
    qdir = Path(out) / "extra" / "quant" / "volume"
    base = {
        split: np.maximum(np.load(Path(out) / split / "volume" / "snaive24.npy"), 0).astype(float)
        for split in evs
    }
    point = {
        split: np.maximum(
            np.load(Path(out) / "extra" / "floor" / "volume" / f"blend_{split}.npy"), 0
        ).astype(float)
        for split in evs
    }
    raw = {
        split: np.maximum(
            np.sort(np.stack([np.load(qdir / f"q{a:.8f}_{split}.npy") for a in alphas]), axis=0), 0
        )
        for split in evs
    }
    calibrated = {split: np.load(qdir / f"calibrated_{split}.npy") for split in evs}
    scaled = {}
    factors = []
    rows = []
    daily = []
    contrasts = []
    triage = []
    active_rows = []
    for level in ("zone_hour", "city_hour"):
        A = evs["val"].A if level == "zone_hour" else evs["val"].A.sum(-1)
        for policy, source in [("seasonal_naive_scaled", base), ("ML_point_scaled", point)]:
            P = source["val"] if level == "zone_hour" else source["val"].sum(-1)
            Pt = source["test"] if level == "zone_hour" else source["test"].sum(-1)
            for ratio in RATIOS:
                hold_prediction = np.empty_like(P)
                for fold, (a, b) in enumerate(FOLDS, 1):
                    past = evs["val"].days < a
                    hold = (evs["val"].days >= a) & (evs["val"].days <= b)
                    k = fit_safety_factor(A[past], P[past], ratio) if past.any() else 1.0
                    hold_prediction[hold] = k * P[hold]
                    factors.append(
                        dict(
                            split="val",
                            level=level,
                            policy=policy,
                            cost_ratio=ratio,
                            fold=fold,
                            k=k,
                            fitting_days=int(past.sum()),
                            fitting_start_day=int(evs["val"].days[past][0]) if past.any() else None,
                            fitting_end_day=int(evs["val"].days[past][-1]) if past.any() else None,
                            evaluation_start_day=a,
                            evaluation_end_day=b,
                            fallback=not past.any(),
                            label="forward validation",
                        )
                    )
                scaled["val", level, policy, ratio] = hold_prediction
                k = fit_safety_factor(A, P, ratio)
                scaled["test", level, policy, ratio] = k * Pt
                factors.append(
                    dict(
                        split="test",
                        level=level,
                        policy=policy,
                        cost_ratio=ratio,
                        fold=5,
                        k=k,
                        fitting_days=111,
                        fitting_start_day=30,
                        fitting_end_day=140,
                        evaluation_start_day=141,
                        evaluation_end_day=180,
                        fallback=False,
                        label="final test",
                    )
                )
    for ratio in RATIOS:
        ev = evs["val"]
        bc = asymmetric_cost(ev.A, base["val"], ratio)
        mc = asymmetric_cost(ev.A, point["val"], ratio)
        gain = 1 - mc.sum((0, 1)) / np.maximum(bc.sum((0, 1)), 1e-12)
        wins = np.zeros(Z, int)
        for a, b in FOLDS:
            mask = (ev.days >= a) & (ev.days <= b)
            wins += mc[mask].sum((0, 1)) < bc[mask].sum((0, 1))
        chosen = (gain >= 0.05) & (wins >= 3)
        for z, zone in enumerate(D["zones"]):
            triage.append(
                dict(
                    cost_ratio=ratio,
                    zone=zone,
                    selected=bool(chosen[z]),
                    validation_relative_cost_gain=float(gain[z]),
                    folds_won=int(wins[z]),
                )
            )
        forward = base["val"].copy()
        for a, b in FOLDS:
            past = ev.days < a
            mask = (ev.days >= a) & (ev.days <= b)
            if past.any():
                g = 1 - mc[past].sum((0, 1)) / np.maximum(bc[past].sum((0, 1)), 1e-12)
                previous_wins = np.zeros(Z, int)
                for pa, pb in FOLDS:
                    if pb < a:
                        pm = (ev.days >= pa) & (ev.days <= pb)
                        previous_wins += mc[pm].sum((0, 1)) < bc[pm].sum((0, 1))
                eligible = (g >= 0.05) & (previous_wins >= 3)
                forward[mask] = np.where(
                    eligible[None, None, :], point["val"][mask], base["val"][mask]
                )
        active = np.any(forward != base["val"], axis=(1, 2))
        for split, ev in evs.items():
            qi = alphas.index(ratio / (1 + ratio))
            policies = {
                "seasonal_naive_point": base[split],
                "ML_point": point[split],
                "ML_raw_quantile": raw[split][qi],
                "ML_calibrated_quantile": calibrated[split][qi],
                "triage_ML_point": np.where(chosen[None, None, :], point[split], base[split]),
            }
            if split == "val":
                policies["forward_triage_ML_point"] = forward
            label = (
                "final test"
                if split == "test"
                else "validation conditional on selected models"
            )
            for level in ("zone_hour", "city_hour"):
                A = ev.A if level == "zone_hour" else ev.A.sum(-1)
                pp = {
                    name: p if level == "zone_hour" else p.sum(-1) for name, p in policies.items()
                }
                for policy in ("seasonal_naive_scaled", "ML_point_scaled"):
                    pp[policy] = scaled[split, level, policy, ratio]
                losses = {
                    name: asymmetric_cost(A, p, ratio).reshape(len(A), -1).sum(1)
                    for name, p in pp.items()
                }
                baseline = losses["seasonal_naive_point"]
                subsets = {
                    "all_validation_111" if split == "val" else "final_test_40": np.ones(
                        len(A), bool
                    )
                }
                if split == "val":
                    subsets["common_folds_2_to_4_81"] = ev.days >= 60
                for name, day in losses.items():
                    for subset, mask in subsets.items():
                        rows.append(
                            dict(
                                split=split,
                                level=level,
                                cost_ratio=ratio,
                                optimal_alpha=ratio / (1 + ratio),
                                policy=name,
                                subset=subset,
                                **cost_summary(day[mask], baseline[mask], seed),
                                label=label,
                                selected_zones=(
                                    int(chosen.sum()) if name == "triage_ML_point" else None
                                ),
                                selected_validation_volume_share=(
                                    float(evs["val"].A[:, :, chosen].sum() / evs["val"].A.sum())
                                    if name == "triage_ML_point"
                                    else None
                                ),
                            )
                        )
                    for d, c, r in zip(ev.days, day, baseline):
                        daily.append(
                            dict(
                                split=split,
                                level=level,
                                cost_ratio=ratio,
                                policy=name,
                                day=int(d),
                                daily_cost_units=float(c),
                                baseline_cost_units=float(r),
                                label=label,
                            )
                        )
                for subset, mask in subsets.items():
                    baseline_sub = baseline[mask]
                    for model, reference in (
                        ("ML_raw_quantile", "seasonal_naive_scaled"),
                        ("ML_raw_quantile", "ML_point_scaled"),
                        ("ML_point_scaled", "seasonal_naive_scaled"),
                    ):
                        diff = (losses[model] - losses[reference])[mask]
                        for block in (1, 7):
                            ix = bootstrap_indices(len(diff), block, seed=seed)
                            draw = 100 * diff[ix].sum(1) / baseline_sub[ix].sum(1)
                            contrasts.append(
                                dict(
                                    split=split,
                                    level=level,
                                    cost_ratio=ratio,
                                    subset=subset,
                                    model=model,
                                    reference=reference,
                                    block_days=block,
                                    difference_percentage_points=float(
                                        100 * diff.sum() / baseline_sub.sum()
                                    ),
                                    lower_percentage_points=float(np.quantile(draw, 0.025)),
                                    upper_percentage_points=float(np.quantile(draw, 0.975)),
                                    label=label,
                                )
                            )
                if split == "val":
                    for subset, mask in [
                        ("all_111_with_zero_padding", np.ones(len(A), bool)),
                        ("prediction_active_days_only", active),
                    ]:
                        if mask.sum() >= 7:
                            active_rows.append(
                                dict(
                                    cost_ratio=ratio,
                                    level=level,
                                    subset=subset,
                                    active_days=int(active.sum()),
                                    total_validation_days=111,
                                    **cost_summary(
                                        losses["forward_triage_ML_point"][mask],
                                        baseline[mask],
                                        seed,
                                    ),
                                    label="validation only; one active regime; zero-padded upper endpoints are not substantive bounds",
                                )
                            )
    complete = write_table(out, "decisions/full_policy_costs", rows)
    write_table(
        out,
        "decision_costs",
        complete[complete.subset.isin(["all_validation_111", "final_test_40"])],
    )
    write_table(out, "decisions/raw_vs_scaled_contrasts", contrasts)
    write_table(out, "decisions/safety_factors", factors)
    write_table(out, "decisions/daily_policy_losses", daily)
    write_table(out, "triage_zone_selection", triage)
    write_table(out, "decisions/forward_triage_active_evaluation", active_rows)
    save_json(
        Path(out) / "decision_assumptions.json",
        dict(
            cost_formula="r*max(A-Q,0)+max(Q-A,0)",
            cost_ratios=RATIOS,
            c_o=1,
            triage_margin=0.05,
            triage_min_folds_won=3,
            validation_factors="earlier folds only; first fold k=1",
            test_factors="validation only; final test",
            city="separate factor fitted to city-hour sums; marginal quantile sums are not city quantiles",
            units="hypothetical relative cost units, not money",
        ),
    )
