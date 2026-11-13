"""Shared state and drawing helpers for the figure modules."""
import hashlib
import json
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..config import SEED
from .style import STYLE, configure_style


def build_context(data, out, evidence):
    configure_style()
    out = Path(out)
    source = Path(evidence)
    figdir = out / "figures"
    res = out / "results"
    figdir.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    V = np.asarray(data["V"]).reshape(-1, 24, len(data["zones"]))
    cc = np.asarray(data["st"]["cc"], float)
    O = np.asarray(data["O"]).reshape(V.shape) / cc
    zones = [str(z) for z in data["zones"]]
    nd = len(V)
    z = len(zones)
    dates = pd.date_range("2022-09-01", periods=nd, freq="D")
    days = {"val": np.arange(30, 141), "test": np.arange(141, nd)}
    arrays = {"volume": V, "occ": O}
    weights = {"volume": np.ones(z), "occ": cc}
    target_name = {"volume": "Estimated energy", "occ": "Occupancy"}
    created = []
    provenance = {}
    loaded = {}
    axis_records = []
    text_records = []
    holidays = [
        ("2022-09-10", "2022-09-12"),
        ("2022-10-01", "2022-10-07"),
        ("2022-12-31", "2023-01-02"),
        ("2023-01-21", "2023-01-27"),
    ]
    makeups = [("2022-10-08", "2022-10-09"), ("2023-01-28", "2023-01-29")]
    rest = (dates.dayofweek >= 5).copy()
    for a, b in holidays:
        rest |= (dates >= a) & (dates <= b)
    for a, b in makeups:
        rest[(dates >= a) & (dates <= b)] = False

    def read(name):
        p = source / (name + ".csv")
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def prediction(target, split, model="blend_training_floor"):
        key = (target, split, model)
        if key not in loaded:
            if model == "blend_training_floor":
                p = source / "extra" / "floor" / target / f"blend_{split}.npy"
            else:
                p = source / split / target / (model + ".npy")
            loaded[key] = np.load(p)
        return loaded[key]

    def figure(height=9, rows=1, cols=1, **kw):
        return plt.subplots(rows, cols, figsize=(STYLE["width_cm"] / 2.54, height / 2.54), **kw)

    def footnote(fig, note, y=0.012, width=95):
        fig.text(
            0.02,
            y,
            textwrap.fill(note, width=width),
            fontsize=8.2,
            color=STYLE["muted"],
            va="bottom",
        )

    def top_legend(fig, handles, ncol=None, y=0.995, size=8.2, x=0.5):
        return fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(x, y),
            frameon=False,
            ncol=ncol or len(handles),
            fontsize=size,
            handlelength=1.8,
            columnspacing=1.5,
            handletextpad=0.5,
        )

    def swatch(colour, label):
        return Patch(facecolor=colour, edgecolor=STYLE["swatch_edge"], linewidth=0.6, label=label)

    def line_handle(colour, label, ls="-", lw=1.6):
        return Line2D([], [], color=colour, ls=ls, lw=lw, label=label)

    def marker_handle(colour, label, marker="o", hollow=False):
        return Line2D(
            [],
            [],
            marker=marker,
            ls="",
            markersize=5,
            markeredgecolor=colour,
            markerfacecolor="white" if hollow else colour,
            label=label,
        )

    def shade_holidays(ax):
        for a, b in holidays:
            ax.axvspan(
                pd.Timestamp(a),
                pd.Timestamp(b) + pd.Timedelta(days=1),
                facecolor=STYLE["holiday"],
                lw=0,
                zorder=1,
            )

    def shade_final(ax):
        ax.axvspan(
            dates[141], dates[-1] + pd.Timedelta(days=1), facecolor=STYLE["test"], lw=0, zorder=0
        )

    def save(name, fig, rows):
        f = pd.DataFrame(rows)
        p = res / ("figure_" + name + ".csv")
        f.to_csv(p, index=False)
        fig.canvas.draw()
        for axis_number, ax in enumerate(fig.axes):
            for axis, values, labels in [
                ("x", ax.get_xticks(), ax.get_xticklabels()),
                ("y", ax.get_yticks(), ax.get_yticklabels()),
            ]:
                for ti, (value, label) in enumerate(zip(values, labels)):
                    axis_records.append(
                        dict(
                            figure=name,
                            axis=axis_number,
                            coordinate=axis,
                            tick_index=ti,
                            value=float(value),
                            displayed_text=label.get_text(),
                            source="Plot coordinate protocol",
                        )
                    )
            labels = [
                ("title", ax.title),
                ("x label", ax.xaxis.label),
                ("y label", ax.yaxis.label),
            ] + [("annotation", t) for t in ax.texts]
            for kind, t in labels:
                text_records.append(
                    dict(
                        figure=name,
                        axis=axis_number,
                        kind=kind,
                        text=t.get_text(),
                        font_size=t.get_fontsize(),
                        numeric_tokens=json.dumps(
                            re.findall(r"[-−]?\d[\d,.]*(?:%|\b)", t.get_text())
                        ),
                        bounds=json.dumps(
                            list(
                                t.get_window_extent().transformed(fig.transFigure.inverted()).bounds
                            )
                        ),
                    )
                )
        for ti, t in enumerate(fig.texts):
            text_records.append(
                dict(
                    figure=name,
                    axis=-1,
                    kind="figure text",
                    text=t.get_text(),
                    font_size=t.get_fontsize(),
                    numeric_tokens=json.dumps(re.findall(r"[-−]?\d[\d,.]*(?:%|\b)", t.get_text())),
                    bounds=json.dumps(
                        list(t.get_window_extent().transformed(fig.transFigure.inverted()).bounds)
                    ),
                )
            )
        for legend in fig.legends:
            for t in legend.get_texts():
                text_records.append(
                    dict(
                        figure=name,
                        axis=-1,
                        kind="legend",
                        text=t.get_text(),
                        font_size=t.get_fontsize(),
                        numeric_tokens=json.dumps(
                            re.findall(r"[-−]?\d[\d,.]*(?:%|\b)", t.get_text())
                        ),
                        bounds=json.dumps(
                            list(
                                t.get_window_extent().transformed(fig.transFigure.inverted()).bounds
                            )
                        ),
                    )
                )
        for ext in ("png", "pdf"):
            fig.savefig(
                figdir / (name + "." + ext), dpi=STYLE["dpi"], bbox_inches="tight", pad_inches=0.04
            )
        provenance[name] = {
            "table": p.name,
            "png_sha256": hashlib.sha256((figdir / (name + ".png")).read_bytes()).hexdigest(),
            "pdf_sha256": hashlib.sha256((figdir / (name + ".pdf")).read_bytes()).hexdigest(),
        }
        plt.close(fig)
        created.append(name)

    def split_label(split):
        return "final test" if split == "test" else "validation"

    def date_axis(ax):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    def label_ticks(ax):
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y")

    def city(A, target):
        return (A * weights[target]).sum(-1)

    def blocks(n, seed=None):
        seed = SEED if seed is None else seed
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, max(n - 6, 1), (5000, int(np.ceil(n / 7))))
        return (starts[:, :, None] + np.arange(7)).reshape(5000, -1)[:, :n].clip(max=n - 1)

    def interval_ratio(num, den):
        ix = blocks(len(num))
        v = num[ix].sum(1) / den[ix].sum(1)
        return np.quantile(v, [0.025, 0.975])

    def metric(target, split, model="blend_training_floor"):
        A = arrays[target][days[split]]
        P = prediction(target, split, model)
        B = prediction(target, split, "snaive24")
        mae = np.abs(P - A).mean()
        bmae = np.abs(B - A).mean()
        ca = city(A, target)
        cp = city(P, target)
        cb = city(B, target)
        return dict(
            zh_MAE=mae,
            zh_WAPE=np.abs(P - A).sum() / np.abs(A).sum(),
            city_MAE=np.abs(cp - ca).mean(),
            city_WAPE=np.abs(cp - ca).sum() / np.abs(ca).sum(),
            zone_MAE_skill=1 - mae / bmae,
            city_MAE_skill=1 - np.abs(cp - ca).sum() / np.abs(cb - ca).sum(),
        )

    metrics = read("all_model_metrics")
    paired = read("paired_confidence_intervals")
    rows = []
    dailyrows = []
    corrrows = []
    for target in ("volume", "occ"):
        for split in ("val", "test"):
            A = arrays[target][days[split]]
            for model in ("snaive24", "lgb_xgb_blend", "blend_training_floor"):
                try:
                    P = prediction(target, split, model)
                except FileNotFoundError:
                    raise
                error = (P - A) * weights[target]
                abs_zone = np.abs(error).sum(-1)
                abs_city = np.abs(error.sum(-1))
                zn = abs_zone.sum(1)
                cn = abs_city.sum(1)
                ratio = cn.sum() / zn.sum()
                low, high = interval_ratio(cn, zn)
                E = error.reshape(-1, z)
                vary = E.std(0) > 1e-12
                valid = np.flatnonzero(vary)
                corr = np.corrcoef(E[:, vary], rowvar=False)
                tri = np.triu_indices(len(valid), 1)
                cv = corr[tri]
                rows.append(
                    dict(
                        target=target,
                        split=split,
                        model=model,
                        days=len(A),
                        abs_city_error_total=cn.sum(),
                        sum_abs_zone_error_total=zn.sum(),
                        city_to_abs_zone_ratio=ratio,
                        cancellation_share=1 - ratio,
                        ratio_block7_lower=low,
                        ratio_block7_upper=high,
                        cancellation_block7_lower=1 - high,
                        cancellation_block7_upper=1 - low,
                        mean_pairwise_error_correlation=float(cv.mean()),
                        valid_zones=len(valid),
                        valid_zone_pairs=len(cv),
                        label=split_label(split),
                        units="Estimated kWh" if target == "volume" else "Busy-pile hours",
                        definition="Absolute weighted city error divided by sum of weighted absolute zone errors; equally weighted pairwise Pearson correlations; correlation interval not estimated",
                    )
                )
                for d, a, b in zip(days[split], cn, zn):
                    dailyrows.append(
                        dict(
                            target=target,
                            split=split,
                            model=model,
                            day=int(d),
                            date=str(dates[d].date()),
                            abs_city_error=a,
                            sum_abs_zone_error=b,
                            cancellation_share=1 - a / b,
                            label=split_label(split),
                        )
                    )
                for i, j, c in zip(valid[tri[0]], valid[tri[1]], cv):
                    corrrows.append(
                        dict(
                            target=target,
                            split=split,
                            model=model,
                            zone_a=zones[i],
                            zone_b=zones[j],
                            correlation=c,
                            label=split_label(split),
                        )
                    )
    cancellation = pd.DataFrame(rows)
    cancellation.to_csv(res / "cancellation_summary.csv", index=False)
    pd.DataFrame(dailyrows).to_csv(res / "cancellation_daily.csv", index=False)
    pd.DataFrame(corrrows).to_csv(res / "cancellation_pair_correlations.csv", index=False)

    return SimpleNamespace(
        V=V,
        data=data,
        arrays=arrays,
        axis_records=axis_records,
        blocks=blocks,
        cancellation=cancellation,
        city=city,
        created=created,
        date_axis=date_axis,
        dates=dates,
        days=days,
        figure=figure,
        footnote=footnote,
        holidays=holidays,
        label_ticks=label_ticks,
        line_handle=line_handle,
        makeups=makeups,
        marker_handle=marker_handle,
        metric=metric,
        metrics=metrics,
        nd=nd,
        paired=paired,
        prediction=prediction,
        provenance=provenance,
        read=read,
        res=res,
        rest=rest,
        save=save,
        shade_final=shade_final,
        shade_holidays=shade_holidays,
        source=source,
        split_label=split_label,
        swatch=swatch,
        target_name=target_name,
        text_records=text_records,
        top_legend=top_legend,
        z=z,
        zones=zones,
    )
