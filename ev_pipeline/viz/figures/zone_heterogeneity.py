"""Figure: zone heterogeneity."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import STYLE


def draw(ctx):
    V = ctx.V
    arrays = ctx.arrays
    cluster_map = ctx.cluster_map
    days = ctx.days
    figure = ctx.figure
    footnote = ctx.footnote
    label_ticks = ctx.label_ticks
    marker_handle = ctx.marker_handle
    prediction = ctx.prediction
    profile_names = ctx.profile_names
    save = ctx.save
    split_label = ctx.split_label
    swatch = ctx.swatch
    top_legend = ctx.top_legend
    z = ctx.z
    zones = ctx.zones

    fig, axs = figure(10.6, 2, 3, gridspec_kw={"width_ratios": [1.2, 1, 0.8]})
    chartrows = []
    training_energy = V[:30].mean((0, 1))
    quartile = pd.qcut(pd.Series(training_energy).rank(method="first"), 4, labels=False).to_numpy()
    worse_counts = {}
    tail_counts = {}
    for row, target in enumerate(["volume", "occ"]):
        axs[row, 2].sharey(axs[row, 1])
        skills = {}
        for split in ["val", "test"]:
            A = arrays[target][days[split]]
            P = prediction(target, split)
            B = prediction(target, split, "snaive24")
            den = np.abs(B - A).mean((0, 1))
            skills[split] = 100 * (1 - np.abs(P - A).mean((0, 1)) / den)
        ax = axs[row, 0]
        values = skills["test"]
        worse = int((values < 0).sum())
        tail = int((values < -100).sum())
        worse_counts[target] = worse
        tail_counts[target] = tail
        clipped = np.maximum(values, -99.999)
        counts, edges = np.histogram(clipped, bins=np.arange(-100, 101, 20))
        centres = (edges[:-1] + edges[1:]) / 2
        ax.bar(
            centres,
            counts,
            width=19,
            color=[STYLE["blend"] if c >= 0 else STYLE["raw"] for c in centres],
            edgecolor="white",
            lw=0.5,
        )
        ax.axvline(0, color=STYLE["ink"], lw=1.3)
        ax.set_title(("Energy" if target == "volume" else "Occupancy") + "*")
        ax.set(xlabel="MAE improvement (%)", ylabel="Zones")
        ax.set_xticks([-100, 0, 100])
        label_ticks(ax)
        for split, dx, marker, face in [
            ("val", -0.14, "o", STYLE["blend"]),
            ("test", 0.14, "D", "white"),
        ]:
            for col, groups, labels in [
                (1, quartile, ["Q1", "Q2", "Q3", "Q4"]),
                (
                    2,
                    np.array([cluster_map[s] for s in zones]),
                    [profile_names[k].split("-")[0] for k in sorted(profile_names)],
                ),
            ]:
                ax = axs[row, col]
                order = sorted(np.unique(groups))
                for pos, g in enumerate(order):
                    v = skills[split][groups == g]
                    median = float(np.median(v))
                    q1, q3 = np.quantile(v, [0.25, 0.75])
                    ax.errorbar(
                        pos + dx,
                        median,
                        yerr=[[median - q1], [q3 - median]],
                        fmt=marker,
                        color=STYLE["blend"],
                        markerfacecolor=face,
                        markersize=4.2,
                        lw=1.2,
                        capsize=2,
                        zorder=3,
                    )
                    chartrows.append(
                        dict(
                            target=target,
                            split=split,
                            group_type="Activity quartile" if col == 1 else "Profile cluster",
                            group=int(g),
                            zones=len(v),
                            median_skill_percent=median,
                            zone_p25=q1,
                            zone_p75=q3,
                            label=split_label(split),
                        )
                    )
                ax.axhline(0, color=STYLE["ink"], lw=0.9, zorder=1)
                ax.set_xticks(range(len(order)), labels)
                ax.set_xlim(-0.6, len(order) - 0.4)
                ax.set_title("By training activity" if col == 1 else "By profile")
                if col == 1:
                    ax.set_ylabel("Median improvement (%)")
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
                label_ticks(ax)
        chartrows.extend(
            dict(
                target=target,
                split="test",
                group_type="Histogram",
                zone=zone,
                skill_percent=float(v),
                below_minus100=bool(v < -100),
                negative_zones=worse,
                tail_zones=tail,
                label="final test",
            )
            for zone, v in zip(zones, values)
        )
    top_legend(
        fig,
        [
            marker_handle(STYLE["blend"], "Validation", "o"),
            marker_handle(STYLE["blend"], "Final window*", "D", hollow=True),
            swatch(STYLE["raw"], "Zones worse than yesterday"),
        ],
    )
    fig.subplots_adjust(left=0.085, right=0.975, top=0.87, bottom=0.19, wspace=0.5, hspace=0.80)
    footnote(
        fig,
        f"*Final test window. Zones worse than yesterday in the final window: energy {worse_counts['volume']} of {z}, occupancy {worse_counts['occ']} of {z}; {tail_counts['volume']} and {tail_counts['occ']} zones below −100% remain in the first histogram bin. Markers show the median zone with its interquartile range, not a confidence interval.",
    )
    save("zone_heterogeneity", fig, chartrows)
    yield "zone_heterogeneity"
