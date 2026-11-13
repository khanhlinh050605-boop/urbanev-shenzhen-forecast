"""Figure: price schedule."""
import numpy as np

from ..style import STYLE


def draw(ctx):
    V = ctx.V
    data = ctx.data
    figure = ctx.figure
    footnote = ctx.footnote
    label_ticks = ctx.label_ticks
    save = ctx.save
    zones = ctx.zones

    if "E" in data:
        p = np.asarray(data["E"]).reshape(V.shape)[:, :, 0]
        mean = p.mean(0)
        low = p.min(0)
        high = p.max(0)
        fig, ax = figure(5.8)
        ax.fill_between(range(24), low, high, color=STYLE["blend"], alpha=0.13, lw=0)
        ax.plot(range(24), mean, color=STYLE["blend"], lw=1.7)
        ax.set(
            title=f"Released fee, zone {zones[0]}: mean and observed range",
            xlabel="Hour of day",
            ylabel="Released fee (CNY/kWh)",
        )
        ax.set_xticks([0, 6, 12, 18, 23])
        label_ticks(ax)
        fig.subplots_adjust(left=0.1, right=0.975, top=0.86, bottom=0.27)
        footnote(
            fig,
            "No next-day tariff feature is used; advance availability of the schedule is unverified.",
        )
        save(
            "price_schedule",
            fig,
            [
                dict(
                    hour=h,
                    mean_fee=m,
                    zone=zones[0],
                    minimum_fee=l,
                    maximum_fee=u,
                    label="Descriptive released values; advance availability unverified",
                )
                for h, m, l, u in zip(range(24), mean, low, high)
            ],
        )
        yield "price_schedule"
