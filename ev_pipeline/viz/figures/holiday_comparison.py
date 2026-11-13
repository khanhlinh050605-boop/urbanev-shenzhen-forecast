"""Figure: holiday comparison."""
import numpy as np
import pandas as pd

from ..style import STYLE


def draw(ctx):
    arrays = ctx.arrays
    city = ctx.city
    dates = ctx.dates
    days = ctx.days
    figure = ctx.figure
    footnote = ctx.footnote
    label_ticks = ctx.label_ticks
    line_handle = ctx.line_handle
    makeups = ctx.makeups
    nd = ctx.nd
    prediction = ctx.prediction
    save = ctx.save
    split_label = ctx.split_label
    swatch = ctx.swatch
    top_legend = ctx.top_legend

    fig, axs = figure(9.5, 2, 3, sharey="row")
    chartrows = []
    events = [
        ("National Day", "2022-10-01", 7),
        ("New Year", "2022-12-31", 3),
        ("Spring Festival*", "2023-01-21", 7),
    ]
    for col, (title, start, length) in enumerate(events):
        first = int((pd.Timestamp(start) - dates[0]).days)
        eventdays = np.arange(first - 1, min(first + length + 3, nd))
        split = "test" if first >= 141 else "val"
        eventdays = eventdays[np.isin(eventdays, days[split])]
        for row, target in enumerate(["volume", "occ"]):
            ax = axs[row, col]
            A = arrays[target][eventdays]
            ix = np.searchsorted(days[split], eventdays)
            P = prediction(target, split, "blend_training_floor")[ix]
            B = prediction(target, split, "snaive24")[ix]
            divisor = 1000 / 24 if target == "volume" else 1
            ax.axvspan(-0.4, length - 0.6, facecolor=STYLE["holiday"], lw=0, zorder=0)
            for model, Q, colour, ls in [
                ("Observed", A, STYLE["observed"], "-"),
                ("Training-floor blend", P, STYLE["blend"], "-"),
                ("Yesterday", B, STYLE["raw"], ":"),
            ]:
                q = city(Q, target).mean(1) / divisor
                ax.plot(eventdays - first, q, color=colour, ls=ls, lw=1.6, zorder=3)
                chartrows.extend(
                    dict(
                        event=title,
                        target=target,
                        relative_day=int(d - first),
                        date=str(dates[d].date()),
                        model=model,
                        value=v,
                        label=split_label(split),
                    )
                    for d, v in zip(eventdays, q)
                )
            for a, b in makeups:
                for d in pd.date_range(a, b):
                    at = int((d - pd.Timestamp(start)).days)
                    if -1 <= at < len(eventdays) - 1:
                        ax.axvline(at, color=STYLE["calibrated"], ls=":", lw=1, zorder=2)
            ax.set_xlabel("Days from holiday start")
            ax.set_xticks([-1, 2, 5, 8] if len(eventdays) > 9 else [-1, 1, 3, 5])
            label_ticks(ax)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel("Thousand kWh per day" if target == "volume" else "Mean busy piles")
    top_legend(
        fig,
        [
            line_handle(STYLE["observed"], "Observed"),
            line_handle(STYLE["blend"], "Training-floor blend"),
            line_handle(STYLE["raw"], "Yesterday", ls=":", lw=1.5),
            swatch(STYLE["holiday"], "Official holiday"),
            line_handle(STYLE["calibrated"], "Make-up workday", ls=":", lw=1.1),
        ],
        ncol=3,
        size=8.2,
    )
    fig.subplots_adjust(left=0.1, right=0.975, top=0.82, bottom=0.23, wspace=0.12, hspace=0.5)
    footnote(
        fig,
        "Energy: city total per day. Occupancy: city mean of busy piles over the day. Y-axes are shared within each row. *Spring Festival lies in the final test window.",
    )
    save("holiday_comparison", fig, chartrows)
    yield "holiday_comparison"
