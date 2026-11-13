"""Figure: paired intervals."""
import numpy as np
import pandas as pd

from ..style import STYLE


def draw(ctx):
    arrays = ctx.arrays
    blocks = ctx.blocks
    city = ctx.city
    days = ctx.days
    figure = ctx.figure
    footnote = ctx.footnote
    marker_handle = ctx.marker_handle
    metric = ctx.metric
    paired = ctx.paired
    prediction = ctx.prediction
    read = ctx.read
    save = ctx.save
    target_name = ctx.target_name
    top_legend = ctx.top_legend

    forest = []
    sensitivity = read("sensitivity_summary")
    for target in ("volume", "occ"):
        for level in ("zone_hour", "city_hour"):
            denominator = metric(target, "val", "snaive24")[
                "zh_MAE" if level == "zone_hour" else "city_MAE"
            ]
            for model, ref, title in [
                ("blend_training_floor", "snaive24", "Blend − yesterday"),
                ("blend_training_floor", "ridge_AR", "Blend − Ridge"),
                (
                    "blend_training_floor",
                    "lgb_training_floor" if target == "volume" else "xgb_training_floor",
                    "Blend − runner-up",
                ),
            ]:
                f = (
                    paired[
                        (paired.target == target)
                        & (paired.split == "val")
                        & (paired.model == model)
                        & (paired.reference == ref)
                        & (paired.level == level)
                        & (paired.block_days == 7)
                    ]
                    if not paired.empty
                    else pd.DataFrame()
                )
                if len(f):
                    r = f.iloc[0]
                    forest.append(
                        dict(
                            target=target,
                            comparison=title,
                            level=level,
                            difference=100 * r.difference / denominator,
                            lower=100 * r.lower / denominator,
                            upper=100 * r.upper / denominator,
                            denominator=denominator,
                            label="validation",
                        )
                    )
            for experiment, ref, title in [
                ("training_floor", "original_locked_blend", "Training floor − original"),
                ("neighbours_given_ratio", "without_neighbours", "With − without neighbours"),
                ("volume_11kw", "snaive24", "Capped energy: blend − yesterday"),
            ]:
                f = (
                    sensitivity[
                        (sensitivity.target == target)
                        & (sensitivity.split == "val")
                        & (sensitivity.experiment == experiment)
                        & (sensitivity.reference == ref)
                        & (sensitivity.level == level)
                    ]
                    if not sensitivity.empty
                    else pd.DataFrame()
                )
                if len(f):
                    r = f.iloc[0]
                    denom = denominator
                    if experiment == "volume_11kw":
                        denom = float(
                            r["zh_MAE" if level == "zone_hour" else "city_MAE"] - r.difference
                        )
                    forest.append(
                        dict(
                            target=target,
                            comparison=title,
                            level=level,
                            difference=100 * r.difference / denom,
                            lower=100 * r.lower / denom,
                            upper=100 * r.upper / denom,
                            denominator=denom,
                            label="validation; fixed comparison",
                        )
                    )
                elif experiment == "training_floor":
                    A = arrays[target][days["val"]]
                    original = prediction(target, "val", "lgb_xgb_blend")
                    corrected = prediction(target, "val", "blend_training_floor")
                    if level == "city_hour":
                        A = city(A, target)
                        original = city(original, target)
                        corrected = city(corrected, target)
                    delta = (
                        (np.abs(corrected - A) - np.abs(original - A)).reshape(len(A), -1).mean(1)
                    )
                    mean = float(delta.mean())
                    lo, hi = np.quantile(delta[blocks(len(delta))].mean(1), [0.025, 0.975])
                    forest.append(
                        dict(
                            target=target,
                            comparison=title,
                            level=level,
                            difference=100 * mean / denominator,
                            lower=100 * lo / denominator,
                            upper=100 * hi / denominator,
                            denominator=denominator,
                            label="validation; paired saved predictions",
                        )
                    )
                elif experiment == "neighbours_given_ratio" and level == "city_hour":
                    dl = read("daily_losses")
                    a = dl[
                        (dl.target == target)
                        & (dl.split == "val")
                        & (dl.model == "lgb_training_floor")
                    ].set_index("day")
                    b = dl[
                        (dl.target == target)
                        & (dl.split == "val")
                        & (dl.model == "lgb_ratio_no_neighbours")
                    ].set_index("day")
                    if len(a) and len(b):
                        delta = (a.city_MAE - b.city_MAE).dropna().to_numpy()
                        mean = float(delta.mean())
                        lo, hi = np.quantile(delta[blocks(len(delta))].mean(1), [0.025, 0.975])
                        forest.append(
                            dict(
                                target=target,
                                comparison=title,
                                level=level,
                                difference=100 * mean / denominator,
                                lower=100 * lo / denominator,
                                upper=100 * hi / denominator,
                                denominator=denominator,
                                label="validation; saved paired daily losses",
                            )
                        )
    if forest:
        fig, axs = figure(10, 2, 1, sharex=True)
        for ax, target in zip(axs, ["volume", "occ"]):
            f = pd.DataFrame(forest)
            f = f[f.target == target]
            order = list(dict.fromkeys(f.comparison))
            for level, dy, marker, col in [
                ("zone_hour", -0.14, "o", STYLE["zone"]),
                ("city_hour", 0.14, "s", STYLE["raw"]),
            ]:
                for i, title in enumerate(order):
                    q = f[(f.comparison == title) & (f.level == level)]
                    if len(q):
                        r = q.iloc[0]
                        ax.errorbar(
                            r.difference,
                            i + dy,
                            xerr=[[r.difference - r.lower], [r.upper - r.difference]],
                            fmt=marker,
                            color=col,
                            markersize=4,
                            lw=1.3,
                            capsize=2.2,
                        )
            ax.axvline(0, color=STYLE["ink"], lw=1.3)
            ax.set_yticks(range(len(order)), order)
            ax.invert_yaxis()
            ax.set_title(target_name[target])
            ax.grid(axis="x")
        axs[-1].set_xlabel("MAE difference as % of validation yesterday MAE")
        top_legend(
            fig,
            [
                marker_handle(STYLE["zone"], "Zone-hour", "o"),
                marker_handle(STYLE["raw"], "City-hour", "s"),
            ],
            x=0.6,
        )
        fig.subplots_adjust(left=0.37, right=0.97, top=0.88, bottom=0.2, hspace=0.42)
        footnote(
            fig,
            "Validation; paired 95% seven-day block intervals. The capped-energy row uses its own yesterday scale. Negative values favour the first model.",
        )
        save("paired_intervals", fig, forest)
        yield "paired_intervals"
