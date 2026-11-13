"""Figure: commitment costs."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import STYLE


def draw(ctx):
    footnote = ctx.footnote
    label_ticks = ctx.label_ticks
    read = ctx.read
    save = ctx.save
    top_legend = ctx.top_legend

    costs = read("decisions/full_policy_costs")
    if costs.empty:
        costs = read("decision_costs")
    if not costs.empty:
        fig = plt.figure(figsize=(STYLE["width_cm"] / 2.54, 11.8 / 2.54))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.15, 1],
            left=0.1,
            right=0.975,
            top=0.84,
            bottom=0.2,
            hspace=0.80,
            wspace=0.34,
        )
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        aw = fig.add_subplot(gs[1, :])
        chartrows = []
        styles = {
            "seasonal_naive_scaled": ("Yesterday × k", STYLE["yesterday"], "--", "o"),
            "ML_point": ("Corrected point", STYLE["floor"], ":", "x"),
            "ML_point_scaled": ("Corrected point × k", STYLE["floor"], "-.", "s"),
            "ML_raw_quantile": ("Raw quantile", STYLE["raw"], "-", "^"),
            "ML_calibrated_quantile": ("Calibrated quantile", STYLE["calibrated"], "--", "D"),
        }
        for ax, level, title in zip(
            axs,
            ["zone_hour", "city_hour"],
            [
                "Energy zone-hour loss\n111 validation days",
                "Energy city-sum loss\n111 validation days",
            ],
        ):
            ax.axhline(0, color=STYLE["ink"], lw=0.8)
            for offset, (policy, (name, colour, ls, mark)) in enumerate(styles.items()):
                f = costs[
                    (costs.split == "val") & (costs.level == level) & (costs.policy == policy)
                ]
                if "subset" in f:
                    f = f[f.subset == "all_validation_111"]
                f = f.sort_values("cost_ratio")
                x = np.arange(len(f))
                y = f.relative_cost_difference * 100
                ax.plot(
                    x + (offset - 2) * 0.018,
                    y,
                    color=colour,
                    ls=ls,
                    marker=mark,
                    markersize=3.8,
                    label=name,
                    lw=1.4,
                )
                chartrows.extend(
                    dict(
                        panel="sweep",
                        level=level,
                        policy=policy,
                        cost_ratio=r.cost_ratio,
                        relative_cost_percent=100 * r.relative_cost_difference,
                        label="validation; initial fold k=1",
                    )
                    for r in f.itertuples()
                )
            ax.set_xticks(range(6), ["0.25", "0.5", "1", "2", "4", "9"])
            ax.set(
                xlabel="Shortage / surplus cost ratio", ylabel="Cost vs yesterday (%)", title=title
            )
            label_ticks(ax)
        decomp = read("decisions/report_cost_decomposition")
        if decomp.empty and "subset" in costs:
            d = costs[
                (costs.split == "val")
                & (costs.level == "zone_hour")
                & (costs.cost_ratio == 4)
                & (costs.subset == "common_folds_2_to_4_81")
            ].set_index("policy")
            if {"ML_point", "ML_point_scaled", "ML_raw_quantile"}.issubset(set(d.index)):
                point = float(d.loc["ML_point", "relative_cost_difference"])
                scaled = float(d.loc["ML_point_scaled", "relative_cost_difference"])
                raw = float(d.loc["ML_raw_quantile", "relative_cost_difference"])
                decomp = pd.DataFrame(
                    [
                        dict(
                            point_skill_vs_plain_percentage_points=-100 * point,
                            incremental_point_scaling_percentage_points=100 * (point - scaled),
                            incremental_raw_over_scaled_point_percentage_points=100
                            * (scaled - raw),
                        )
                    ]
                )
        if not decomp.empty:
            d = decomp.iloc[0]
            amounts = [
                d.point_skill_vs_plain_percentage_points,
                d.incremental_point_scaling_percentage_points,
                d.incremental_raw_over_scaled_point_percentage_points,
            ]
            labels = []
            cumulative = 0
            for i, (name, amount, colour) in enumerate(
                zip(
                    ["Corrected point", "Safety-factor scaling", "Quantile increment"],
                    amounts,
                    [STYLE["floor"], "#7BBFB8", STYLE["raw"]],
                )
            ):
                aw.bar(i, amount, bottom=cumulative, color=colour, width=0.56)
                aw.plot(
                    [i + 0.28, i + 0.72],
                    [cumulative + amount] * 2,
                    color=STYLE["muted"],
                    lw=0.8,
                )
                labels.append(f"{name}\n{amount:.2f}")
                chartrows.append(
                    dict(
                        panel="waterfall",
                        component=name,
                        improvement_points=amount,
                        prior_total=cumulative,
                        result_total=cumulative + amount,
                        evaluation_days=81,
                        cost_ratio=4,
                        label="Validation; common later 81 days; path-dependent arithmetic",
                    )
                )
                cumulative += amount
            aw.bar(3, cumulative, color=STYLE["ink"], width=0.56)
            labels.append(f"Total\n{cumulative:.2f}")
            aw.set_xticks(range(4), labels)
            aw.set_ylabel("Cost reduction (points)")
            aw.set_ylim(0, cumulative * 1.08)
            aw.set_title("Decomposition at cost ratio 4, common later 81 validation days")
            label_ticks(aw)
        top_legend(
            fig, [axs[0].get_legend_handles_labels()[0][i] for i in range(5)], ncol=3, size=8.2
        )
        footnote(
            fig,
            "Sweep: 111 validation days (first-fold k=1). Decomposition: common later 81 days. Waterfall units are points of plain-yesterday cost, not causal attribution.",
        )
        save("commitment_costs", fig, chartrows)
        yield "commitment_costs"
