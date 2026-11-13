"""Figure: calibration."""
import numpy as np

from ..style import STYLE


def draw(ctx):
    figure = ctx.figure
    footnote = ctx.footnote
    label_ticks = ctx.label_ticks
    line_handle = ctx.line_handle
    marker_handle = ctx.marker_handle
    read = ctx.read
    save = ctx.save
    split_label = ctx.split_label
    target_name = ctx.target_name
    top_legend = ctx.top_legend

    intervals = read("interval_metrics")
    if not intervals.empty:
        fig, axs = figure(8.2, 2, 2, sharex="col")
        chartrows = []
        for col, target in enumerate(["volume", "occ"]):
            f = intervals[
                (intervals.target == target)
                & (intervals.level == "zone_hour")
                & intervals.method.isin(["raw", "forward_calibrated"])
            ]
            if "subset" in f:
                f = f[f.subset.isna()]
            axs[0, col].axhline(80, color=STYLE["muted"], ls="--", lw=1)
            for method, dx, colr, mark in [
                ("raw", -0.13, STYLE["raw"], "o"),
                ("forward_calibrated", 0.13, STYLE["calibrated"], "s"),
            ]:
                for x, split in enumerate(["val", "test"]):
                    g = f[(f.method == method) & (f.split == split)]
                    if g.empty:
                        continue
                    r = g.iloc[0]
                    axs[0, col].scatter(
                        x + dx, 100 * r.coverage, color=colr, marker=mark, s=30, zorder=3
                    )
                    if np.isfinite(r.get("coverage_block7_lower", np.nan)):
                        axs[0, col].errorbar(
                            x + dx,
                            100 * r.coverage,
                            yerr=[
                                [100 * (r.coverage - r.coverage_block7_lower)],
                                [100 * (r.coverage_block7_upper - r.coverage)],
                            ],
                            fmt="none",
                            color=colr,
                            capsize=2.5,
                            lw=1.2,
                            zorder=2,
                        )
                    axs[1, col].scatter(
                        x + dx, r.mean_width, color=colr, marker=mark, s=30, zorder=3
                    )
                    chartrows.append(
                        dict(
                            target=target,
                            split=split,
                            method=method,
                            coverage=r.coverage,
                            mean_width=r.mean_width,
                            label=split_label(split),
                        )
                    )
            axs[0, col].set(
                title=target_name[target], ylabel="Empirical coverage (%)", ylim=(65, 90)
            )
            axs[1, col].set_ylabel(
                "Mean width (kWh)" if target == "volume" else "Mean width (ratio)"
            )
            for ax in axs[:, col]:
                ax.set_xticks([0, 1], ["Validation", "Final window*"])
                ax.set_xlim(-0.5, 1.5)
                label_ticks(ax)
        top_legend(
            fig,
            [
                marker_handle(STYLE["raw"], "Raw quantiles", "o"),
                marker_handle(STYLE["calibrated"], "Calibrated quantiles", "s"),
                line_handle(STYLE["muted"], "80% nominal coverage", ls="--", lw=1),
            ],
        )
        fig.subplots_adjust(left=0.1, right=0.975, top=0.85, bottom=0.14, hspace=0.3, wspace=0.34)
        footnote(
            fig,
            "Bars on coverage are 95% seven-day block intervals. *Final test window.",
        )
        save("calibration", fig, chartrows)
        yield "calibration"
