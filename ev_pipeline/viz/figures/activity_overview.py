"""Figure: activity overview."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import STYLE


def draw(ctx):
    V = ctx.V
    date_axis = ctx.date_axis
    dates = ctx.dates
    label_ticks = ctx.label_ticks
    rest = ctx.rest
    save = ctx.save
    shade_final = ctx.shade_final
    shade_holidays = ctx.shade_holidays
    swatch = ctx.swatch
    top_legend = ctx.top_legend
    z = ctx.z

    fig = plt.figure(figsize=(STYLE["width_cm"] / 2.54, 10.6 / 2.54))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.12, 1],
        left=0.1,
        right=0.975,
        top=0.86,
        bottom=0.115,
        hspace=0.68,
        wspace=0.34,
    )
    ax = fig.add_subplot(gs[0, :])
    a1 = fig.add_subplot(gs[1, 0])
    a2 = fig.add_subplot(gs[1, 1])
    daily = V.sum((1, 2)) / 1e6
    shade_final(ax)
    shade_holidays(ax)
    ax.plot(dates, daily, color=STYLE["observed"], lw=1.5, zorder=3)
    ax.set_xlim(dates[0] - pd.Timedelta(days=2), dates[-1] + pd.Timedelta(days=2))
    ax.set(title=f"Daily estimated energy, all {z} zones", ylabel="Million kWh per day")
    date_axis(ax)
    label_ticks(ax)
    top_legend(
        fig,
        [
            swatch(STYLE["holiday"], "Official holiday"),
            swatch(STYLE["test"], "Final test window"),
        ],
    )
    chartrows = [
        dict(
            panel="daily",
            date=str(t.date()),
            estimated_kwh=v * 1e6,
            label="descriptive; includes the final test window",
        )
        for t, v in zip(dates, daily)
    ]
    hourly = V.sum(-1)
    for mask, name, color in [
        (~rest, "Working day", STYLE["blend"]),
        (rest, "Rest day", STYLE["raw"]),
    ]:
        p = hourly[mask].mean(0) / 1000
        a1.plot(range(24), p, color=color, label=name, lw=1.7)
        chartrows.extend(
            dict(
                panel="profile",
                hour=h,
                day_type=name,
                thousand_kwh=v,
                label="descriptive; includes final window",
            )
            for h, v in enumerate(p)
        )
    a1.set(title="Average hourly profile", xlabel="Hour of day", ylabel="Thousand kWh per hour")
    a1.set_xticks([0, 6, 12, 18, 23])
    a1.legend(
        frameon=False,
        fontsize=8.2,
        loc="lower left",
        bbox_to_anchor=(0, 1.21),
        ncol=2,
        borderaxespad=0,
    )
    label_ticks(a1)
    energy = np.sort(V.sum((0, 1)))[::-1]
    share = energy.cumsum() / energy.sum() * 100
    a2.plot(np.arange(1, z + 1) / z * 100, share, color=STYLE["blend"], lw=1.7)
    a2.set(
        title="Concentration across zones",
        xlabel="Zones ranked by activity (%)",
        ylabel="Cumulative energy (%)",
    )
    a2.set_ylim(0, 102)
    label_ticks(a2)
    chartrows.extend(
        dict(
            panel="concentration",
            rank=i + 1,
            zone_percent=(i + 1) / z * 100,
            energy_percent=v,
            label="descriptive; includes final window",
        )
        for i, v in enumerate(share)
    )
    save("activity_overview", fig, chartrows)
    yield "activity_overview"
