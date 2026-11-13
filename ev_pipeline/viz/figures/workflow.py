"""Figure: workflow."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from ..style import STYLE


def draw(ctx):
    save = ctx.save
    z = ctx.z

    fig = plt.figure(figsize=(STYLE["width_cm"] / 2.54, 8.8 / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()
    chartrows = []
    ax.text(
        50,
        95.5,
        "Forecast origin 23:00 on day T  →  hourly forecast for every zone on day T+1",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#2F3B4D",
    )
    stages = [
        (
            "1  Data",
            "#0072B2",
            "UrbanEV hourly panel\n%d zones, 181 days\nvolume and occupancy" % z,
        ),
        ("2  Cleaning", "#009E73", "Release audit\nvolume = kWh estimate\nratio = busy / piles"),
        (
            "3  Features",
            "#E69F00",
            "Daily and weekly lags\nlagged price and weather\ncalendar, neighbour summaries",
        ),
        (
            "4  Splits",
            "#7B5EA7",
            "111-day rolling validation\n40-day final window\n(final test)",
        ),
        (
            "5  Models",
            "#C0392B",
            "LightGBM + XGBoost blend\noriginal + training floors\nvs yesterday baselines",
        ),
        (
            "6  Evaluation",
            "#2F3B4D",
            "MAE, WAPE, skill\npaired block bootstrap\nzone and city level",
        ),
    ]
    width, height, header = 30.5, 28, 9
    xs = [1.5, 34.75, 68.0]
    ys = [60, 24]
    for i, (title, colour, text) in enumerate(stages):
        row, col = divmod(i, 3)
        x, y = xs[col], ys[row]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=2.4",
                facecolor="white",
                edgecolor=colour,
                lw=1.8,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, y + height - header),
                width,
                header,
                boxstyle="round,pad=0,rounding_size=2.4",
                facecolor=colour,
                edgecolor=colour,
                lw=1.8,
            )
        )
        ax.add_patch(
            Rectangle(
                (x, y + height - header), width, 3.5, facecolor=colour, edgecolor=colour, lw=1.8
            )
        )
        ax.text(
            x + width / 2,
            y + height - header / 2,
            title,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="white",
        )
        ax.text(
            x + width / 2,
            y + (height - header) / 2,
            text,
            ha="center",
            va="center",
            fontsize=8.2,
            linespacing=1.55,
            color="#222B38",
        )
        chartrows.append(dict(order=i + 1, stage=title, content=text.replace("\n", "; ")))
    arrow = dict(arrowstyle="-|>", lw=1.6, color="#6B7480", mutation_scale=9, shrinkA=0, shrinkB=0)
    for row in (0, 1):
        for col in (0, 1):
            y = ys[row] + height / 2
            ax.annotate(
                "", xy=(xs[col + 1] - 0.8, y), xytext=(xs[col] + width + 0.8, y), arrowprops=arrow
            )
    ax.plot(
        [xs[2] + width / 2] * 2, [ys[0] - 0.2, 56], color="#6B7480", lw=1.6, solid_capstyle="butt"
    )
    ax.plot(
        [xs[2] + width / 2, xs[0] + width / 2],
        [56, 56],
        color="#6B7480",
        lw=1.6,
        solid_capstyle="butt",
    )
    ax.annotate(
        "",
        xy=(xs[0] + width / 2, ys[1] + height + 0.6),
        xytext=(xs[0] + width / 2, 56),
        arrowprops=arrow,
    )
    ax.annotate(
        "",
        xy=(xs[2] + width / 2, 14.6),
        xytext=(xs[2] + width / 2, ys[1] - 0.2),
        arrowprops=arrow,
    )
    ax.add_patch(
        FancyBboxPatch(
            (1.5, 3),
            97,
            11,
            boxstyle="round,pad=0,rounding_size=2.4",
            facecolor="#F3F5F7",
            edgecolor="#9AA3AE",
            lw=1.6,
        )
    )
    last = "7  Decision analysis: asymmetric costs, safety factor k, quantile commitments, holiday analysis"
    ax.text(50, 8.5, last, ha="center", va="center", fontsize=8.1, color="#222B38")
    chartrows.append(dict(order=7, stage="Decision analysis", content=last[3:]))
    save("workflow", fig, chartrows)
    yield "workflow"
