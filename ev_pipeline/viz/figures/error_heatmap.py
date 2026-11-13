"""Figure: error heatmap."""
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle

from ..style import STYLE


def draw(ctx):
    arrays = ctx.arrays
    dates = ctx.dates
    days = ctx.days
    figure = ctx.figure
    footnote = ctx.footnote
    holidays = ctx.holidays
    line_handle = ctx.line_handle
    prediction = ctx.prediction
    rest = ctx.rest
    save = ctx.save
    split_label = ctx.split_label
    swatch = ctx.swatch
    target_name = ctx.target_name
    top_legend = ctx.top_legend

    fig, axs = figure(12.4, 2, 1)
    chartrows = []
    cmap = LinearSegmentedColormap.from_list(
        "blend_vs_yesterday", ["#1F5C99", "#FFFFFF", "#E08A00"]
    )
    for ax, target in zip(axs, ["volume", "occ"]):
        full = []
        ds = []
        for split in ["val", "test"]:
            A = arrays[target][days[split]]
            P = prediction(target, split)
            B = prediction(target, split, "snaive24")
            q = (np.abs(P - A) - np.abs(B - A)).mean(-1)
            if target == "occ":
                q *= 100
            full.append(q)
            ds.extend(days[split].tolist())
            chartrows.extend(
                dict(
                    target=target,
                    day=int(d),
                    hour=h,
                    mae_difference=float(v),
                    previous_day_rest=bool(rest[d - 1]),
                    label=split_label(split),
                )
                for d, line in zip(days[split], q)
                for h, v in enumerate(line)
            )
        q = np.concatenate(full)
        ds = np.asarray(ds)
        first = int(np.where(ds >= 141)[0][0])
        edges = sorted(set([0] + list(range(first, 0, -7))))
        edges += list(range(first + 7, len(ds), 7)) + [len(ds)]
        bins = list(zip(edges[:-1], edges[1:]))
        weekly = np.stack([q[a:b].mean(0) for a, b in bins], axis=1)
        lim = float(np.abs(weekly).max())
        im = ax.imshow(
            weekly,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
            interpolation="nearest",
        )
        ticks = list(range(0, len(bins), 3))
        ax.set_xticks(ticks, [dates[ds[bins[i][0]]].strftime("%d %b") for i in ticks])
        ax.set_yticks([0, 6, 12, 18, 23])
        ax.set_ylabel("Target hour")
        ax.set_title(target_name[target], pad=22)
        ax.set_xlabel("First day of the seven-day bin")
        boundary = [i for i, (a, b) in enumerate(bins) if a == first][0] - 0.5
        ax.axvline(boundary, color=STYLE["ink"], ls="--", lw=1.3)
        for i, (a, b) in enumerate(bins):
            span = dates[ds[a:b]]
            if any(
                ((span >= pd.Timestamp(h0)) & (span <= pd.Timestamp(h1))).any()
                for h0, h1 in holidays
            ):
                ax.add_patch(
                    Rectangle(
                        (i - 0.5, 1.02),
                        1,
                        0.07,
                        transform=ax.get_xaxis_transform(),
                        facecolor=STYLE["holiday"],
                        edgecolor=STYLE["raw"],
                        lw=0.6,
                        clip_on=False,
                    )
                )
        ax.add_patch(
            Rectangle(
                (boundary, 1.11),
                len(bins) - boundary - 0.5,
                0.07,
                transform=ax.get_xaxis_transform(),
                facecolor=STYLE["test"],
                edgecolor="#B9A8D2",
                lw=0.6,
                clip_on=False,
            )
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.018, fraction=0.045, extend="neither")
        cbar.ax.tick_params(labelsize=8.2)
        cbar.set_label("kWh" if target == "volume" else "Ratio, percentage points", fontsize=8.2)
        cbar.set_ticks([-lim, 0, lim])
        cbar.outline.set_visible(False)
    top_legend(
        fig,
        [
            swatch(STYLE["holiday"], "Bin contains an official holiday"),
            swatch(STYLE["test"], "Final test window"),
            line_handle(STYLE["ink"], "Final window starts", ls="--", lw=1),
        ],
    )
    fig.subplots_adjust(left=0.1, right=0.86, top=0.84, bottom=0.17, hspace=0.95)
    footnote(
        fig,
        "Blend minus yesterday MAE, averaged over seven-day bins for each target hour; negative (blue) favours the blend. Colour limits are symmetric around zero and no cell is clipped.",
    )
    save("error_heatmap", fig, chartrows)
    yield "error_heatmap"
