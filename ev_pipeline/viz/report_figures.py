"""Report figures 3, 4, 7, 8, 9 and 10 drawn at the printed text width (160 mm), from results/*.csv.

Every figure is inserted in the report at width=\textwidth (1:1), so fonts are ~6.5 pt.
Usage: python -m ev_pipeline.viz.report_figures output/results output/figures
"""
import sys
from pathlib import Path
import cnsplots as cns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from . import merged as m
from .merged import OI, TN, FS, _read, _hforest, _dates

RES = Path("output/results")
OUT = Path("output/figures")
PW = 160  # mm, printed text width


def new():
    cns.settings.axes_titlelocation = "left"
    return cns.multipanel(max_width=PW, unit="mm")


def panel(mp, lab, w, h, **kw):
    kw.setdefault("margin_right", 3)
    kw.setdefault("margin_bottom", 3)
    return mp.panel(lab, width=w, height=h, **kw)


def fit(mp):
    """Set tick / label sizes to the printed size (the figure is inserted at 100 %)."""
    for ax in mp.fig.axes:
        ax.tick_params(labelsize=FS, length=2, pad=1.5)
        ax.xaxis.label.set_size(FS + 0.3)
        ax.yaxis.label.set_size(FS + 0.3)


def save(mp, name):
    fit(mp)
    w, h = mp.fig.get_size_inches() * 25.4
    print(name, "size mm: %.1f x %.1f" % (w, h))
    cns.savefig(str(OUT / (name + ".png")), fig=mp.fig)
    cns.savefig(str(OUT / (name + ".pdf")), fig=mp.fig)


def fig10():
    fp = _read(RES / "decisions", "full_policy_costs")
    fp = fp[(fp.split == "val") & (fp.subset == "common_folds_2_to_4_81")]
    ct = _read(RES / "decisions", "raw_vs_scaled_contrasts")
    ct = ct[(ct.split == "val") & (ct.subset == "common_folds_2_to_4_81") & (ct.level == "zone_hour")
            & (ct.cost_ratio == 4) & (ct.block_days == 7)]
    im = _read(RES, "interval_metrics")
    mp = new()
    sty = {"seasonal_naive_scaled": ("Yesterday × k", OI["orange"], "--"), "ML_point": ("ML point", OI["sky"], ":"),
           "ML_point_scaled": ("ML point × k", OI["blue"], "-."), "ML_raw_quantile": ("Raw quantile", OI["purple"], "-"),
           "ML_calibrated_quantile": ("Calibrated quantile", OI["grey"], "--")}
    H = 40
    for lab, lv, w, padl in [("A", "zone_hour", 27, 9), ("B", "city_hour", 25, 8)]:
        ax = panel(mp, lab, w, H, pad_left=padl, pad_top=7, margin_bottom=19)
        ax.axhline(0, color=OI["black"], lw=0.6)
        for pol, (name, c, ls) in sty.items():
            r = fp[(fp.level == lv) & (fp.policy == pol)].sort_values("cost_ratio")
            ax.plot(range(len(r)), 100 * r.relative_cost_difference, ls, color=c, marker="o", ms=2.2, lw=1.0)
        ax.set_xticks(range(6), ["¼", "½", "1", "2", "4", "9"])
        ax.set_xlabel("Cost ratio")
        ax.set_title(("Zone-hour" if lv == "zone_hour" else "City-hour") + ", 81 days", fontsize=FS)
        if lab == "A":
            ax.set_ylabel("Cost vs plain yesterday (%)")
        else:
            ax.set_ylim(*axA_ylim) if False else None
    # C: accounting path at ratio 4
    z = fp[(fp.level == "zone_hour") & (fp.cost_ratio == 4)].set_index("policy")
    adj_y = -100 * z.loc["seasonal_naive_scaled", "relative_cost_difference"]
    adj_ml = -100 * z.loc["ML_point_scaled", "relative_cost_difference"]
    raw_q = -100 * z.loc["ML_raw_quantile", "relative_cost_difference"]
    g = ct.set_index(["model", "reference"])
    steps = [("Yesterday × k", adj_y, OI["orange"]), ("ML beyond it", adj_ml - adj_y, OI["blue"]), ("Quantile", raw_q - adj_ml, OI["purple"])]
    ci = [(-100 * z.loc["seasonal_naive_scaled", "relative_block7_upper"], -100 * z.loc["seasonal_naive_scaled", "relative_block7_lower"]),
          (-g.loc[("ML_point_scaled", "seasonal_naive_scaled"), "upper_percentage_points"], -g.loc[("ML_point_scaled", "seasonal_naive_scaled"), "lower_percentage_points"]),
          (-g.loc[("ML_raw_quantile", "ML_point_scaled"), "upper_percentage_points"], -g.loc[("ML_raw_quantile", "ML_point_scaled"), "lower_percentage_points"])]
    ax = panel(mp, "C", 25, H, pad_left=9, pad_top=7, margin_bottom=19)
    cum_ = 0
    for i, ((name, v, col), (lo, hi)) in enumerate(zip(steps, ci)):
        ax.bar(i, v, bottom=cum_, color=col, width=0.62)
        ax.plot([i, i], [cum_ + lo, cum_ + hi], color=OI["black"], lw=0.8)
        ax.text(i, cum_ + v + 1.2 if False else cum_ + hi + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=FS - 1)
        cum_ += v
    ax.bar(len(steps), cum_, color=OI["black"], width=0.62)
    ax.text(len(steps), cum_ + 1.0, f"{cum_:.1f}", ha="center", va="bottom", fontsize=FS - 1)
    ax.set_xticks(range(len(steps) + 1), ["Yest. × k", "ML beyond", "Quantile", "Total"], rotation=35, ha="right")
    ax.set_ylabel("Cost reduction (points)")
    ax.set_ylim(0, 47)
    ax.set_title("Ratio 4, 81 days", fontsize=FS)
    # D: coverage dots
    ax = panel(mp, "D", 27, H, pad_left=9, pad_top=7, margin_bottom=19, margin_right=0)
    ticks, tl, x = [], [], 0
    for t, c in [("volume", OI["blue"]), ("occ", OI["orange"])]:
        for s in ["val", "test"]:
            for k, (mth, filled) in enumerate([("raw", False), ("forward_calibrated", True)]):
                r = im[(im.target == t) & (im.split == s) & (im.method == mth) & (im.level == "zone_hour")].iloc[0]
                xx = x + (k - 0.5) * 0.3
                ax.errorbar(xx, 100 * r.coverage, yerr=[[100 * (r.coverage - r.coverage_block7_lower)], [100 * (r.coverage_block7_upper - r.coverage)]],
                            fmt="o", color=c, markerfacecolor=c if filled else "white", ms=3.0, lw=0.9, capsize=1.2)
            ticks.append(x); tl.append("val." if s == "val" else "final")
            x += 1
    ax.axhline(80, color=OI["black"], ls="--", lw=0.8)
    ax.set_xlim(-0.6, x - 0.4)
    ax.set_xticks(ticks, tl)
    for xc, nm in [(0.5, "Energy"), (2.5, "Occupancy")]:
        ax.text(xc, -0.17, nm, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS + 0.3)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("80% interval", fontsize=FS)
    # shared legend in its own row
    mp.newline()
    lax = panel(mp, " ", 150, 4, pad_left=2, pad_top=0, margin_bottom=0, margin_right=0)
    lax.axis("off")
    hs = [Line2D([0], [0], color=c, ls=ls, marker="o", ms=2.2, lw=1.0, label=n) for n, c, ls in sty.values()]
    hs += [Line2D([0], [0], marker="o", ls="", color=OI["black"], markerfacecolor="white", ms=3, label="Raw (D)"),
           Line2D([0], [0], marker="o", ls="", color=OI["black"], markerfacecolor=OI["black"], ms=3, label="Calibrated (D)")]
    lax.legend(handles=hs, loc="center", ncol=7, frameon=False, fontsize=FS - 0.5, handlelength=1.6, columnspacing=0.8, borderpad=0)
    save(mp, "fig10")


def fig7():
    zh = _read(RES, "figure_zone_heterogeneity")
    profiles = _read(RES, "cluster_profiles"); profiles.columns = profiles.columns.map(str)
    pn = m._profile_names(profiles)
    mp = new()
    H = 36
    for lab, t, w, padl in [("A", "volume", 28, 9), ("B", "occ", 28, 8)]:
        ax = panel(mp, lab, w, H, pad_left=padl, pad_top=7, margin_bottom=14)
        v = zh[(zh.target == t) & (zh.group_type == "Histogram")].skill_percent.to_numpy()
        counts, edges = np.histogram(v[v >= -100], bins=np.arange(-100, 101, 20))
        c = (edges[:-1] + edges[1:]) / 2
        ax.bar(c, counts, width=19, color=[OI["blue"] if k >= 0 else OI["orange"] for k in c], edgecolor="white", lw=0.4)
        ax.bar(-120, int((v < -100).sum()), width=19, color="white", edgecolor=OI["orange"], lw=0.7, hatch="////")
        ax.axvline(-108, color="0.55", lw=0.6, ls=(0, (2, 2)))
        ax.axvline(0, color=OI["black"], lw=0.8)
        ax.set_xlim(-132, 105)
        ax.set_xticks([-120, -50, 0, 50, 100], ["<−100", "−50", "0", "50", "100"])
        ax.set_xlabel("MAE improvement (%)")
        ax.set_title(TN[t] + ", final window", fontsize=FS)
        if lab == "A":
            ax.set_ylabel("Zones")
    series = [("volume", "val", OI["blue"], "o", True), ("volume", "test", OI["blue"], "D", False),
              ("occ", "val", OI["orange"], "o", True), ("occ", "test", OI["orange"], "D", False)]
    for lab, gt, labels, w, padl, last in [("C", "Activity quartile", ["Q1", "Q2", "Q3", "Q4"], 36, 9, False),
                                           ("D", "Profile cluster", [pn[k].split("-")[0] for k in sorted(pn)], 22, 8, True)]:
        ax = panel(mp, lab, w, H, pad_left=padl, pad_top=7, margin_bottom=14, margin_right=0 if last else 3)
        ax.axhline(0, color=OI["black"], lw=0.7)
        for i, (t, s_, c, mk, filled) in enumerate(series):
            d = zh[(zh.target == t) & (zh.split == s_) & (zh.group_type == gt)].sort_values("group")
            xx = np.arange(len(d)) + (i - 1.5) * 0.17
            ax.errorbar(xx, d.median_skill_percent, yerr=[d.median_skill_percent - d.zone_p25, d.zone_p75 - d.median_skill_percent],
                        fmt=mk, color=c, markerfacecolor=c if filled else "white", ms=3.0, lw=0.9, capsize=1.2)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(-50, 40); ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_title("By training activity" if lab == "C" else "By profile group", fontsize=FS)
        if lab == "C":
            ax.set_ylabel("Median improvement (%)")
    mp.newline()
    lax = panel(mp, " ", 150, 4, pad_left=2, pad_top=0, margin_bottom=0, margin_right=0)
    lax.axis("off")
    hs = [Line2D([0], [0], marker=mk, ls="", color=c, markerfacecolor=c if filled else "white", ms=3.4,
                 label=f"{TN[t]}, {'validation' if s_ == 'val' else 'final'}") for t, s_, c, mk, filled in series]
    lax.legend(handles=hs, loc="center", ncol=4, frameon=False, fontsize=FS - 0.3, handletextpad=0.3, columnspacing=1.6, borderpad=0)
    save(mp, "fig7")


def fig8():
    tests = _read(RES, "zone_failure_tests"); conc = _read(RES, "zone_failure_concentration")
    traj = _read(RES, "zone_failure_trajectory"); abl = _read(RES, "ablation_steps"); feat = _read(RES, "figure_feature_gain")
    names = {"zero_share": "Zero-hour share", "festival_ratio": "Festival ratio", "max_daily_jump_final": "Largest daily jump", "test_vs_last30": "Level vs last 30 d"}
    mp = new()
    H1, H2 = 30, 25
    ax = panel(mp, "A", 30, H1, pad_left=32, pad_top=6, margin_bottom=25)
    labs, est, lo, hi, cols = [], [], [], [], []
    for t, c in [("volume", OI["blue"]), ("occ", OI["orange"])]:
        for k in names:
            r = tests[(tests.target == t) & (tests.covariate == k)].iloc[0]
            labs.append(f"{names[k]} ({TN[t][0]})"); est.append(r.auc); lo.append(r.auc_lower); hi.append(r.auc_upper); cols.append(c)
    _hforest(ax, labs, est, lo, hi, cols, ref=0.5)
    ax.set_xlabel("AUC, losing vs other zones")
    ax.set_title("Who loses", fontsize=FS)
    ax = panel(mp, "B", 25, H1, pad_left=23, pad_top=6, margin_bottom=25)
    a = abl[(abl.split == "test") & (abl.target == "volume")]
    a = a[a.step.str.startswith("Total") == False]
    labs = ["Add neighbours", "Ratio target", "Training floor", "XGBoost blend"]
    _hforest(ax, labs, a.percent, a.percent_lower, a.percent_upper, [OI["blue"]] * len(a))
    ax.set_xlabel("Energy MAE change (%)")
    ax.set_title("Design steps", fontsize=FS)
    ax = panel(mp, "C", 24, H1, pad_left=10, pad_top=6, margin_bottom=25, margin_right=0)
    for t, c in [("volume", OI["blue"]), ("occ", OI["orange"])]:
        d = conc[conc.target == t]
        ax.plot(range(len(d)), 100 * d.excess_share, "o-", color=c, ms=2.8, lw=1.0, label=TN[t] if t == "volume" else "Occup.")
    ax.set_xticks(range(4), ["1", "5", "10", "51"])
    ax.set_xlabel("Worst losing zones")
    ax.set_ylabel("Share of excess error (%)")
    ax.legend(frameon=False, fontsize=FS - 0.5, loc="lower right", handlelength=1.2, bbox_to_anchor=(1.04, -0.02), labelspacing=0.3, handletextpad=0.4)
    ax.set_title("Concentration", fontsize=FS)
    mp.newline()
    ax = panel(mp, "D", 85, H2, pad_left=10, pad_top=8, margin_bottom=12)
    for g, ls in [("losing", "-"), ("other", "--")]:
        for s_, c in [("observed", OI["black"]), ("blend", OI["blue"]), ("yesterday", OI["orange"])]:
            d = traj[(traj.target == "volume") & (traj.group == g) & (traj.series == s_)]
            ax.plot(pd.to_datetime(d.date), d.relative_level, ls, color=c, lw=0.9)
    ax.set_ylabel("Level vs last 30 d")
    ax.set_ylim(0.3, 1.25)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    _dates(ax, "%d %b")
    ax.set_title("Energy: solid = 51 losing zones, dashed = other 224", fontsize=FS)
    hs = [Line2D([0], [0], color=OI["black"], lw=1.0, label="Observed"), Line2D([0], [0], color=OI["blue"], lw=1.0, label="Blend"),
          Line2D([0], [0], color=OI["orange"], lw=1.0, label="Yesterday")]
    ax.legend(handles=hs, frameon=False, fontsize=FS - 0.5, ncol=3, loc="lower right", columnspacing=0.8, handlelength=1.5)
    ax = panel(mp, "E", 46, H2, pad_left=10, pad_top=8, margin_bottom=12, margin_right=0)
    cols = {"Own level / history": OI["blue"], "Neighbour / city": OI["orange"], "Other features": OI["grey"]}
    for k, t in enumerate(["volume", "occ"]):
        left = 0
        for gname, c in cols.items():
            v = float(feat[(feat.target == t) & (feat.feature_group == gname)].gain_share.iloc[0]) * 100
            ax.bar(k, v, bottom=left, color=c, width=0.55, label=gname if k == 0 else None)
            if v > 6:
                ax.text(k, left + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=FS - 0.5, color="white" if c != OI["grey"] else "black")
            left += v
    ax.set_xticks([0, 1], ["Energy", "Occup."])
    ax.set_ylabel("Split gain (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, 3.1)
    ax.spines["bottom"].set_bounds(-0.6, 1.6)
    ax.legend(frameon=False, fontsize=FS - 0.5, loc="center right", bbox_to_anchor=(1.03, 0.5), handlelength=1.0)
    ax.set_title("LightGBM split gain", fontsize=FS)
    save(mp, "fig8")


SH = m.SHORT; MD = m.MODEL


def fig3():
    acc, ws = _read(RES, "figure_main_accuracy"), _read(RES, "figure_window_skill")
    mp = new()
    ax = panel(mp, "A", 68, 38, pad_left=10, pad_top=6, margin_bottom=17, margin_right=4)
    order = list(MD); x = np.arange(len(order))
    for k, (t, c) in enumerate([("volume", OI["blue"]), ("occ", OI["orange"])]):
        for j, (s, a) in enumerate([("val", 0.55), ("test", 1.0)]):
            d = acc[(acc.target == t) & (acc.split == s)].set_index("model").reindex(order)
            ax.bar(x + (k * 2 + j - 1.5) * 0.2, d.wape_percent, 0.19, color=c, alpha=a,
                   label=f"{m.TN[t]}, {'validation' if s == 'val' else 'final test'}")
    ax.set_xticks(x, [SH[o] for o in order], rotation=35, ha="right")
    ax.set_ylabel("WAPE (%)"); ax.set_ylim(0, 60)
    ax.legend(frameon=False, fontsize=FS - 0.5, ncol=2, loc="upper left", columnspacing=0.8, handlelength=1.0)
    ax = panel(mp, "B", 68, 38, pad_left=11, pad_top=6, margin_bottom=17, margin_right=0)
    wins = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Test"]
    for t, c in [("volume", OI["blue"]), ("occ", OI["orange"])]:
        for lv, ls in [("Zone-hour", "-"), ("City-hour", "--")]:
            d = ws[(ws.target == t) & (ws.level == lv)].set_index("window").reindex(wins)
            ax.plot(range(5), d.mae_skill_percent, ls, marker="o", ms=2.5, lw=1.1, color=c, label=f"{m.TN[t]}, {lv.lower()}")
    ax.axvline(3.5, color=OI["grey"], ls=":", lw=0.8)
    ax.axhline(0, color=OI["black"], lw=0.5)
    ax.set_xticks(range(5), wins)
    ax.set_ylabel("MAE skill vs yesterday (%)")
    ax.set_ylim(-36, 38); ax.set_yticks([-20, -10, 0, 10, 20, 30])
    ax.legend(frameon=False, fontsize=FS - 0.5, ncol=1, loc="lower left", handlelength=1.6, labelspacing=0.25)
    save(mp, "fig3")


def fig4():
    fc = _read(RES, "figure_forecasts"); smp = _read(RES, "figure_representative_zones")
    ticks = pd.to_datetime(["2023-01-24", "2023-02-07", "2023-02-21"])
    def fmt(ax):
        ax.set_xticks(ticks); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    mp = new()
    for lab, t, yl, pl, mr in [("A", "volume", "Energy (million kWh/day)", 11, 4), ("B", "occ", "Mean busy piles", 13, 0)]:
        ax = panel(mp, lab, 62, 32, pad_left=pl, pad_top=7, margin_bottom=10, margin_right=mr)
        d = fc[fc.target == t]; x = pd.to_datetime(d.date)
        ax.axvspan(pd.Timestamp("2023-01-21"), pd.Timestamp("2023-01-28"), color=OI["yellow"], alpha=0.35, lw=0)
        ax.plot(x, d.observed, color=OI["black"], lw=1.1, label="Observed")
        ax.plot(x, d.training_floor_blend, color=OI["blue"], lw=1.1, label="Blend")
        ax.plot(x, d.yesterday, color=OI["orange"], lw=1.0, ls=":", label="Yesterday")
        ax.set_ylabel(yl); ax.set_title(m.TN[t] + ", city mean", fontsize=FS); fmt(ax)
        if lab == "A":
            ax.legend(frameon=False, fontsize=FS - 0.5, loc="lower right", handlelength=1.4)
    mp.newline()
    for lab, role, pl, mr in [("C", "Median winning zone", 11, 2), ("D", "Median losing zone", 9, 2), ("E", "Worst losing zone", 8, 0)]:
        ax = panel(mp, lab, 40, 30, pad_left=pl, pad_top=7, margin_bottom=10, margin_right=mr)
        d = smp[smp.role == role]; x = pd.to_datetime(d.date)
        ax.axvspan(pd.Timestamp("2023-01-21"), pd.Timestamp("2023-01-28"), color=OI["yellow"], alpha=0.35, lw=0)
        ax.plot(x, d.observed_kwh, color=OI["black"], lw=1.1)
        ax.plot(x, d.blend_kwh, color=OI["blue"], lw=1.1)
        ax.plot(x, d.yesterday_kwh, color=OI["orange"], lw=1.0, ls=":")
        ax.set_title(f"{role} ({int(d.zone.iloc[0])}; skill {str(round(d.skill_percent.iloc[0])).replace('-', chr(8722))}%)", fontsize=FS - 0.3)
        fmt(ax)
        if lab == "C": ax.set_ylabel("Zone energy (kWh/day)")
        if lab == "E":
            ax.set_ylim(-0.06, 1.05)
    save(mp, "fig4")


def fig9():
    h = _read(RES, "figure_holiday_comparison")
    sty = {"Observed": (OI["black"], "-"), "Training-floor blend": (OI["blue"], "-"), "Yesterday": (OI["orange"], ":")}
    events = ["National Day", "New Year", "Spring Festival*"]
    span = {"volume": 0.85, "occ": 1450}
    mp = new(); labs = iter("ABCDEF")
    for row, t in enumerate(["volume", "occ"]):
        for col, ev in enumerate(events):
            ax = panel(mp, next(labs), 40, 27, pad_left=13 if col == 0 else 10, pad_top=7, margin_bottom=10, margin_right=2 if col < 2 else 0)
            d = h[(h.event == ev) & (h.target == t)]
            sc = 1000 if t == "volume" else 1
            lo = hi = None
            for mm, (c, ls) in sty.items():
                r = d[d.model == mm]
                ax.plot(r.relative_day, r.value / sc, color=c, ls=ls, lw=1.1, label=mm)
            v = d.value / sc
            mid = (v.max() + v.min()) / 2
            ax.set_ylim(mid - span[t] / 2, mid + span[t] / 2)
            ax.set_xticks([0, 2, 4] if col == 1 else [0, 2, 4, 6, 8])
            ax.set_title(ev.replace("*", " (final test)") if row == 0 else "", fontsize=FS)
            if col == 0:
                ax.set_ylabel("Energy (million kWh/day)" if t == "volume" else "Busy piles (mean)")
            if row == 1:
                ax.set_xlabel("Day relative to first holiday day")
            if row == 0 and col == 2:
                ax.legend(frameon=False, fontsize=FS - 1.2, loc="upper left", handlelength=1.4)
        if row == 0:
            mp.newline()
    save(mp, "fig9")


def main(argv=None):
    """Usage: python -m ev_pipeline.viz.report_figures [results_dir] [out_dir]"""
    global RES, OUT
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 0:
        RES = Path(argv[0])
    if len(argv) > 1:
        OUT = Path(argv[1])
    OUT.mkdir(exist_ok=True, parents=True)
    for f in (fig3, fig4, fig7, fig8, fig9, fig10):
        f()


if __name__ == "__main__":
    main()
