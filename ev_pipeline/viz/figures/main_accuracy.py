"""Figure: main accuracy."""
from ..style import STYLE


def draw(ctx):
    figure = ctx.figure
    marker_handle = ctx.marker_handle
    metric = ctx.metric
    metrics = ctx.metrics
    save = ctx.save
    source = ctx.source
    split_label = ctx.split_label
    target_name = ctx.target_name
    top_legend = ctx.top_legend

    models = [
        "snaive24",
        "snaive168",
        "hour_dow_mean",
        "ridge_AR",
        "lgb_base",
        "lgb_nbr_ratio",
        "xgb",
        "blend_training_floor",
    ]
    names = {
        "snaive24": "Yesterday",
        "snaive168": "Last week",
        "hour_dow_mean": "Hour / weekday mean",
        "ridge_AR": "Ridge autoregression",
        "lgb_base": "LightGBM",
        "lgb_nbr_ratio": "Scaled LightGBM + neighbours",
        "xgb": "XGBoost",
        "blend_training_floor": "Training-floor blend",
    }
    fig, axs = figure(8.2, 1, 2, sharey=True)
    chartrows = []
    for ax, target in zip(axs, ["volume", "occ"]):
        available = []
        for model in models:
            if not metrics.empty and ((metrics.target == target) & (metrics.model == model)).any():
                available.append(model)
            elif (source / "val" / target / (model + ".npy")).exists():
                available.append(model)
        for i, model in enumerate(available):
            values = []
            for split, dy, mark in [("val", -0.14, "o"), ("test", 0.14, "D")]:
                if (
                    metrics.empty
                    or not (
                        (metrics.target == target)
                        & (metrics.model == model)
                        & (metrics.split == split)
                    ).any()
                ):
                    v = metric(target, split, model)["zh_WAPE"] * 100
                else:
                    v = (
                        float(
                            metrics[
                                (metrics.target == target)
                                & (metrics.model == model)
                                & (metrics.split == split)
                            ]
                            .iloc[0]
                            .zh_WAPE
                        )
                        * 100
                    )
                colour = (
                    STYLE["blend"]
                    if model == "blend_training_floor"
                    else (STYLE["raw"] if model == "snaive24" else "#8794A6")
                )
                ax.scatter(
                    v,
                    i + dy,
                    marker=mark,
                    s=26,
                    edgecolors=colour,
                    facecolors=colour if split == "val" else "white",
                    linewidths=1.2,
                    zorder=3,
                )
                chartrows.append(
                    dict(
                        target=target,
                        split=split,
                        model=model,
                        wape_percent=v,
                        label=split_label(split),
                    )
                )
                values.append(v)
            ax.plot(values, [i - 0.14, i + 0.14], color="#CBD3DD", lw=1.1, zorder=1)
        ax.set_yticks(range(len(available)), [names[n] for n in available])
        ax.set(xlabel="Zone-hour WAPE (%)", title=target_name[target])
        ax.grid(axis="x")
    axs[0].invert_yaxis()
    for tick in axs[0].get_yticklabels():
        if tick.get_text() == "Training-floor blend":
            tick.set_fontweight("bold")
    top_legend(
        fig,
        [
            marker_handle(STYLE["muted"], "Validation", "o"),
            marker_handle(STYLE["muted"], "Final test window", "D", hollow=True),
        ],
        x=0.6,
    )
    fig.subplots_adjust(left=0.3, right=0.97, top=0.86, bottom=0.17, wspace=0.12)
    save("main_accuracy", fig, chartrows)
    yield "main_accuracy"
