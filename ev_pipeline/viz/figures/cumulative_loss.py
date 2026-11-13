"""Figure: cumulative loss."""
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from ..style import STYLE


def draw(ctx):
    arrays = ctx.arrays
    city = ctx.city
    date_axis = ctx.date_axis
    dates = ctx.dates
    days = ctx.days
    figure = ctx.figure
    footnote = ctx.footnote
    label_ticks = ctx.label_ticks
    prediction = ctx.prediction
    save = ctx.save
    shade_final = ctx.shade_final
    swatch = ctx.swatch
    target_name = ctx.target_name
    top_legend = ctx.top_legend

    fig, axs = figure(9.6, 2, 2)
    chartrows = []
    for i, target in enumerate(["volume", "occ"]):
        ds = []
        dz = []
        dc = []
        for split in ["val", "test"]:
            A = arrays[target][days[split]]
            P = prediction(target, split)
            B = prediction(target, split, "snaive24")
            dz.extend((np.abs(P - A).mean((1, 2)) - np.abs(B - A).mean((1, 2))).tolist())
            dc.extend(
                (
                    np.abs(city(P, target) - city(A, target)).mean(1)
                    - np.abs(city(B, target) - city(A, target)).mean(1)
                ).tolist()
            )
            ds.extend(days[split].tolist())
        for j, (delta, level, col) in enumerate(
            [(dz, "Zone-hour", STYLE["zone"]), (dc, "City-hour", STYLE["raw"])]
        ):
            ax = axs[i, j]
            cumulative = np.cumsum(delta)
            xx = dates[ds]
            shade_final(ax)
            ax.axhline(0, color=STYLE["ink"], lw=0.8)
            ax.plot(xx, cumulative, color=col, lw=1.6, zorder=3)
            ax.set_xlim(xx[0], xx[-1] + pd.Timedelta(days=1))
            unit = (
                "kWh" if target == "volume" else ("ratio" if level == "Zone-hour" else "busy piles")
            )
            ax.set_title(target_name[target] + " · " + level)
            ax.set_ylabel("Cumulative change\n(" + unit + ")")
            date_axis(ax)
            label_ticks(ax)
            ax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda v, p: (f"{v:,.0f}" if abs(v) >= 10 else f"{v:g}").replace("-", "−")
                )
            )
            chartrows.extend(
                dict(
                    target=target,
                    level=level,
                    date=str(t.date()),
                    daily_difference=d,
                    cumulative_difference=c,
                    label="final test" if day >= 141 else "validation",
                )
                for t, day, d, c in zip(xx, ds, delta, cumulative)
            )
    top_legend(fig, [swatch(STYLE["test"], "Final test window")])
    fig.subplots_adjust(left=0.135, right=0.975, top=0.84, bottom=0.13, hspace=0.55, wspace=0.55)
    footnote(fig, "Cumulative MAE change, blend minus yesterday; lower is better.")
    save("cumulative_loss", fig, chartrows)
    yield "cumulative_loss"
