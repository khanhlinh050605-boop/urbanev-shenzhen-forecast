"""Figure: feature gain."""
import json

import pandas as pd

from ..style import STYLE


def draw(ctx):
    figure = ctx.figure
    footnote = ctx.footnote
    read = ctx.read
    save = ctx.save
    source = ctx.source

    gain = read("feature_gain_summary")
    if gain.empty:
        rows = []
        own = {f"l{x}" for x in [24, 48, 72, 96, 120, 144, 168]} | {
            "cut0",
            "cut1",
            "cut2",
            "dm1",
            "dm_same",
            "dmax1",
            "dm7",
            "hr7",
            "hr7std",
            "hd_mean",
            "log_s",
        }
        for target in ["volume", "occ"]:
            path = source / "extra" / "floor" / target / "gain_seed42_trainend140.json"
            if not path.exists():
                path = source / "importance" / f"{target}_training_floor_141_180.json"
            if path.exists():
                g = json.loads(path.read_text())
                total = sum(g.values())
                rows.append(
                    dict(
                        target=target,
                        own_level_gain_share=sum(v for k, v in g.items() if k in own) / total,
                        neighbour_city_gain_share=sum(
                            v for k, v in g.items() if k.startswith(("nb_", "city_"))
                        )
                        / total,
                    )
                )
        gain = pd.DataFrame(rows)
    if not gain.empty:
        fig, ax = figure(5.8)
        chartrows = []
        side = []
        for i, target in enumerate(["volume", "occ"]):
            r = gain[gain.target == target].iloc[0]
            values = [
                r.own_level_gain_share,
                r.neighbour_city_gain_share,
                1 - r.own_level_gain_share - r.neighbour_city_gain_share,
            ]
            left = 0
            for value, name, colour in zip(
                values,
                ["Own level / history", "Neighbour / city", "Other features"],
                [STYLE["blend"], STYLE["floor"], "#CBD3DD"],
            ):
                ax.barh(
                    i,
                    value * 100,
                    left=left * 100,
                    color=colour,
                    height=0.52,
                    edgecolor="white",
                    lw=0.8,
                    label=name if i == 0 else None,
                )
                chartrows.append(dict(target=target, feature_group=name, gain_share=value))
                left += value
            side.append(" · ".join(f"{round(100 * v)}%" for v in values))
        ax.set_yticks([0, 1], ["Estimated energy", "Occupancy"])
        ax.set_ylim(1.55, -0.55)
        ax.set_xlabel("Share of LightGBM split gain (%)")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x")
        twin = ax.twinx()
        twin.set_ylim(ax.get_ylim())
        twin.set_yticks([0, 1], side)
        twin.tick_params(length=0, labelsize=8.2, labelcolor=STYLE["ink"])
        for spine in twin.spines.values():
            spine.set_visible(False)
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.45, 0.995),
            frameon=False,
            ncol=3,
            fontsize=8.2,
            handlelength=1.5,
        )
        fig.subplots_adjust(left=0.19, right=0.8, top=0.86, bottom=0.35)
        footnote(
            fig,
            "Gain shares are listed at the right in the order own history, neighbour or city, other. Gain is not causal importance.",
        )
        save("feature_gain", fig, chartrows)
        yield "feature_gain"
