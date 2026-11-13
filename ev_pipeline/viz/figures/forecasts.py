"""Figure: forecasts."""
import matplotlib.dates as mdates
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
    prediction = ctx.prediction
    save = ctx.save
    swatch = ctx.swatch
    target_name = ctx.target_name
    top_legend = ctx.top_legend

    fig, axs = figure(10.8, 2, 1)
    chartrows = []
    for ax, target in zip(axs, ["volume", "occ"]):
        A = arrays[target][days["test"]]
        P = prediction(target, "test")
        B = prediction(target, "test", "snaive24")
        ac = city(A, target)
        pc = city(P, target)
        bc = city(B, target)
        divisor = 1e6 / 24 if target == "volume" else 1
        observed = ac.mean(1) / divisor
        pred = pc.mean(1) / divisor
        base = bc.mean(1) / divisor
        mae = np.abs(pc - ac).mean(1) / divisor
        xx = dates[days["test"]]
        ax.axvspan(
            pd.Timestamp("2023-01-21"),
            pd.Timestamp("2023-01-28"),
            facecolor=STYLE["holiday"],
            lw=0,
            zorder=0,
        )
        ax.fill_between(
            xx, pred - mae, pred + mae, color=STYLE["blend"], alpha=0.12, linewidth=0, zorder=1
        )
        ax.plot(xx, observed, color=STYLE["observed"], lw=1.6, zorder=3)
        ax.plot(xx, pred, color=STYLE["blend"], lw=1.6, zorder=3)
        ax.plot(xx, base, color=STYLE["raw"], ls=":", lw=1.5, zorder=2)
        ax.set_title(target_name[target])
        ax.set_ylabel("Million kWh per day" if target == "volume" else "Mean busy piles")
        label_ticks(ax)
        ax.set_xlim(xx[0], xx[-1])
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        chartrows.extend(
            dict(
                target=target,
                date=str(t.date()),
                observed=a,
                training_floor_blend=p,
                yesterday=b,
                daily_mae_band=m,
                label="final test",
            )
            for t, a, p, b, m in zip(xx, observed, pred, base, mae)
        )
    top_legend(
        fig,
        [
            line_handle(STYLE["observed"], "Observed"),
            line_handle(STYLE["blend"], "Training-floor blend"),
            line_handle(STYLE["raw"], "Yesterday", ls=":", lw=1.5),
            swatch("#DCE6F2", "Daily error band"),
            swatch(STYLE["holiday"], "Spring Festival holiday"),
        ],
        ncol=3,
        size=8.2,
    )
    fig.subplots_adjust(left=0.1, right=0.975, top=0.83, bottom=0.14, hspace=0.45)
    footnote(
        fig,
        "Final test window. The band shows the scaled daily mean hourly absolute error, not a prediction interval.",
    )
    save("forecasts", fig, chartrows)
    yield "forecasts"
