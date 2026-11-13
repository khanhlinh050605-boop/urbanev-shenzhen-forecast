"""Figure: window skill."""
import numpy as np

from ..style import STYLE


def draw(ctx):
    arrays = ctx.arrays
    city = ctx.city
    days = ctx.days
    figure = ctx.figure
    footnote = ctx.footnote
    label_ticks = ctx.label_ticks
    marker_handle = ctx.marker_handle
    nd = ctx.nd
    prediction = ctx.prediction
    save = ctx.save
    split_label = ctx.split_label
    swatch = ctx.swatch
    target_name = ctx.target_name
    top_legend = ctx.top_legend

    fig, axs = figure(7.4, 1, 2)
    chartrows = []
    windows = [
        ("Fold 1", 30, 59),
        ("Fold 2", 60, 89),
        ("Fold 3", 90, 119),
        ("Fold 4", 120, 140),
        ("Test", 141, nd - 1),
    ]
    for ax, target in zip(axs, ["volume", "occ"]):
        ax.axvspan(3.5, 4.5, facecolor=STYLE["test"], lw=0, zorder=0)
        ax.axhline(0, color=STYLE["ink"], lw=0.8)
        for level, key, col, marker in [
            ("Zone-hour", "zone_MAE_skill", STYLE["zone"], "o"),
            ("City-hour", "city_MAE_skill", STYLE["raw"], "s"),
        ]:
            values = []
            for title, a, b in windows:
                split = "test" if a >= 141 else "val"
                mask = (days[split] >= a) & (days[split] <= b)
                A = arrays[target][days[split][mask]]
                P = prediction(target, split)[mask]
                B = prediction(target, split, "snaive24")[mask]
                if level == "City-hour":
                    A = city(A, target)
                    P = city(P, target)
                    B = city(B, target)
                value = 100 * (1 - np.abs(P - A).sum() / np.abs(B - A).sum())
                values.append(value)
                chartrows.append(
                    dict(
                        target=target,
                        window=title,
                        level=level,
                        mae_skill_percent=value,
                        label=split_label(split),
                    )
                )
            ax.plot(range(5), values, color=col, marker=marker, markersize=4.5, lw=1.7, zorder=3)
        ax.set_xticks(range(5), ["1", "2", "3", "4", "Test"])
        ax.set_xlim(-0.4, 4.5)
        ax.set(
            xlabel="Frozen evaluation window",
            ylabel="MAE improvement vs yesterday (%)",
            title=target_name[target],
        )
        label_ticks(ax)
    top_legend(
        fig,
        [
            marker_handle(STYLE["zone"], "Zone-hour", "o"),
            marker_handle(STYLE["raw"], "City-hour", "s"),
            swatch(STYLE["test"], "Final test window"),
        ],
    )
    fig.subplots_adjust(left=0.11, right=0.975, top=0.84, bottom=0.25, wspace=0.3)
    footnote(fig, "Model settings are held fixed across windows.")
    save("window_skill", fig, chartrows)
    yield "window_skill"
