"""Multi-panel figures built with cnsplots from the saved figure tables.

Every figure combines results that answer one question, and every panel reads a CSV in results/
(the representative-zone table is written here from the saved predictions, then plotted from the CSV).
Layout: 183 mm maximum width, Okabe-Ito colours, panel labels A, B, C, ...
"""
from pathlib import Path

import cnsplots as cns
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#8A8F98")
W = 183  # mm
TN = {"volume": "Energy", "occ": "Occupancy"}
MODEL = {"snaive24": "Yesterday", "snaive168": "Same weekday last week", "hour_dow_mean": "Hour × weekday mean",
         "ridge_AR": "Ridge", "lgb_base": "LightGBM", "lgb_nbr_ratio": "LightGBM + ratio", "xgb": "XGBoost",
         "blend_training_floor": "Blend"}
SHORT = {"snaive24": "Yesterday", "snaive168": "Last week", "hour_dow_mean": "Hour×day", "ridge_AR": "Ridge",
         "lgb_base": "LGBM", "lgb_nbr_ratio": "LGBM+ratio", "xgb": "XGB", "blend_training_floor": "Blend"}
HOLIDAY_SPANS = [("2022-09-10", "2022-09-12"), ("2022-10-01", "2022-10-07"),
            ("2022-12-31", "2023-01-02"), ("2023-01-21", "2023-01-27")]
FINAL_START = pd.Timestamp("2023-01-20")  # first day of the final test window (day 141)
FS = 6.5  # base font size in points; every figure is inserted at 100 %


def _read(res, name):
    return pd.read_csv(res / (name + ".csv"))


def _new():
    cns.settings.axes_titlelocation = "left"
    return cns.multipanel(max_width=W, unit="mm")


def _panel(mp, label, w, h, **kw):
    kw.setdefault("margin_right", 6)
    kw.setdefault("margin_bottom", 6)
    return mp.panel(label, width=w, height=h, **kw)


def _save(mp, out, name):
    cns.savefig(str(out / (name + ".png")), fig=mp.fig)
    cns.savefig(str(out / (name + ".pdf")), fig=mp.fig)
    return name


def _hforest(ax, labels, est, lo, hi, colors, ref=0.0):
    y = np.arange(len(labels))[::-1]
    ax.axvline(ref, color=OI["black"], lw=0.7)
    for yi, e, l, h, c in zip(y, est, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=1.6)
        ax.plot(e, yi, "o", color=c, ms=4)
    ax.set_yticks(y, labels)
    ax.spines[["top", "right"]].set_visible(False)


def _dates(ax, fmt="%b"):
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def _shade(ax, final=True, holidays=True):
    if holidays:
        for a, b in HOLIDAY_SPANS:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1), color=OI["yellow"], alpha=0.35, lw=0)
    if final:
        ax.axvline(FINAL_START, color=OI["grey"], ls=":", lw=0.9)


def _profile_names(profiles):
    names = {}
    for g, f in profiles.groupby("cluster"):
        peak = int(f[[str(h) for h in range(24)]].mean().to_numpy().argmax())
        names[int(g)] = "Night-peaked" if peak < 8 or peak >= 22 else "Day-peaked"
    return names


# --------------------------------------------------------------------------- data overview
def overview(res, out, data_dir):
    """Figure 2: activity (A-C) and the training-period profile groups (D-E)."""
    import shapefile
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Patch, Polygon
    from pyproj import CRS, Transformer

    ao = _read(res, "figure_activity_overview")
    clusters = _read(res, "zone_clusters")
    profiles = _read(res, "cluster_profiles")
    profiles.columns = profiles.columns.map(str)
    names = _profile_names(profiles)
    ccol = {g: (OI["blue"] if n == "Day-peaked" else OI["orange"]) for g, n in names.items()}
    mp = _new()
    ax = _panel(mp, "A", 100, 34, pad_left=14, pad_top=6, margin_bottom=24)
    d = ao[ao.panel == "daily"]
    ax.axvspan(FINAL_START, pd.Timestamp("2023-02-28"), color="#E3E5EA", lw=0, zorder=0)
    for h0, h1 in HOLIDAY_SPANS:
        ax.axvspan(pd.Timestamp(h0), pd.Timestamp(h1) + pd.Timedelta(days=1), color="#F5EE9E", lw=0, zorder=1)
    ax.plot(pd.to_datetime(d.date), d.estimated_kwh / 1e6, color=OI["black"], lw=1.1)
    ax.set_xlim(pd.Timestamp("2022-08-30"), pd.Timestamp("2023-02-28"))
    ax.set_ylabel("Energy (million kWh/day)")
    ax.set_title("Daily energy, all zones")
    _dates(ax, "%b")
    ax = _panel(mp, "B", 44, 34, pad_left=14, pad_top=6, margin_bottom=24)
    d = ao[ao.panel == "profile"]
    for name, c in [("Working day", OI["blue"]), ("Rest day", OI["orange"])]:
        r = d[d.day_type == name]
        ax.plot(r.hour, r.thousand_kwh, color=c, lw=1.4, label=name)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Energy (thousand kWh/h)")
    ax.set_title("Hourly profile")
    ax.legend(frameon=False, fontsize=FS - 0.5, loc="lower left", bbox_to_anchor=(0, 0.02))
    mp.newline()
    ax = _panel(mp, "C", 42, 40, pad_left=14, pad_top=6, margin_bottom=14)
    d = ao[ao.panel == "concentration"]
    ax.plot(d.zone_percent, d.energy_percent, color=OI["blue"], lw=1.4)
    ax.plot([0, 100], [0, 100], color=OI["grey"], ls=":", lw=0.8)
    ax.set_xlabel("Zones ranked by activity (%)")
    ax.set_ylabel("Cumulative energy (%)")
    ax.set_title("Concentration")
    ax.set_ylim(0, 102)
    # map
    ax = _panel(mp, "D", 54, 40, pad_left=14, pad_top=6, margin_bottom=14)
    shp = Path(data_dir) / "SZ_districts" / "SZ_districts.shp"
    reader = shapefile.Reader(str(shp))
    proj = Transformer.from_crs(CRS.from_wkt(shp.with_suffix(".prj").read_text()), 4326, always_xy=True)
    cm = {str(int(z)): int(g) for z, g in zip(clusters.zone, clusters.cluster)}
    patches, cols = [], []
    for item in reader.iterShapeRecords():
        zid = str(int(float(item.record["TAZID"])))
        g = cm.get(zid)
        parts = list(item.shape.parts) + [len(item.shape.points)]
        for a, b in zip(parts[:-1], parts[1:]):
            xy = np.asarray(item.shape.points[a:b])
            lon, lat = proj.transform(xy[:, 0], xy[:, 1])
            patches.append(Polygon(np.column_stack([lon, lat])))
            cols.append(ccol[g] if g is not None else "#E4E8ED")
    ax.add_collection(PatchCollection(patches, facecolors=cols, edgecolors="white", linewidths=0.15))
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.deg2rad(22.6)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Profile groups")
    handles = [Patch(facecolor=ccol[g], label=f"{names[g]} ({int((clusters.cluster == g).sum())})") for g in sorted(names)]
    handles.append(Patch(facecolor="#E4E8ED", edgecolor=OI["grey"], linewidth=0.4, label="Not in sample"))
    ax.legend(handles=handles, frameon=True, framealpha=0.95, edgecolor="white", fontsize=FS - 1, loc="upper right", handlelength=1)
    ax = _panel(mp, "E", 42, 40, pad_left=14, pad_top=6, margin_bottom=14)
    for g, f in profiles.groupby("cluster"):
        pp = f[[str(h) for h in range(24)]].mean().to_numpy()
        ax.plot(range(24), pp, color=ccol[int(g)], lw=1.4)
    ax.axhline(1, color=OI["grey"], ls=":", lw=0.8)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Relative to zone mean")
    ax.set_title("Training profiles")
    return _save(mp, out, "merged_overview")


# --------------------------------------------------------------------------- accuracy
def accuracy(res, out):
    acc, ws = _read(res, "figure_main_accuracy"), _read(res, "figure_window_skill")
    mp = _new()
    ax = _panel(mp, "A", 70, 42, pad_left=14, pad_top=8, margin_bottom=22)
    order = list(MODEL)
    x = np.arange(len(order))
    for k, (t, c) in enumerate([("volume", OI["blue"]), ("occ", OI["orange"])]):
        for j, (s, a) in enumerate([("val", 0.55), ("test", 1.0)]):
            d = acc[(acc.target == t) & (acc.split == s)].set_index("model").reindex(order)
            ax.bar(x + (k * 2 + j - 1.5) * 0.2, d.wape_percent, 0.19, color=c, alpha=a,
                   label=f"{TN[t]}, {'validation' if s == 'val' else 'final test'}")
    ax.set_xticks(x, [SHORT[m] for m in order], rotation=35, ha="right")
    ax.set_ylabel("WAPE (%)")
    ax.set_ylim(0, 58)
    ax.legend(frameon=False, fontsize=FS - 0.5, ncol=2, loc="upper right")
    ax = _panel(mp, "B", 70, 42, pad_left=14, pad_top=8, margin_bottom=22)
    wins = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Test"]
    for t, c in [("volume", OI["blue"]), ("occ", OI["orange"])]:
        for lv, ls in [("Zone-hour", "-"), ("City-hour", "--")]:
            d = ws[(ws.target == t) & (ws.level == lv)].set_index("window").reindex(wins)
            ax.plot(range(5), d.mae_skill_percent, ls, marker="o", ms=3, color=c, label=f"{TN[t]}, {lv.lower()}")
    ax.axvline(3.5, color=OI["grey"], ls=":", lw=0.8)
    ax.set_xticks(range(5), wins)
    ax.set_ylabel("MAE skill vs yesterday (%)")
    ax.set_ylim(-14, 36)
    ax.legend(frameon=False, fontsize=FS - 0.5, ncol=2, loc="lower right")
    return _save(mp, out, "merged_accuracy")


# --------------------------------------------------------------------------- representative zones
def sample_table(res, data):
    """Representative zones (energy, final window). Rule: rank the 275 zones by final-window MAE skill over
    yesterday; show the median winning zone, the median losing zone and the worst losing zone."""
    diag = _read(res, "zone_failure_diagnostics")
    d = diag[diag.target == "volume"].copy()
    wins, loses = d[~d.loses].sort_values("skill"), d[d.loses].sort_values("skill")
    pick = [("Median winning zone", wins.iloc[len(wins) // 2]),
            ("Median losing zone", loses.iloc[len(loses) // 2]),
            ("Worst losing zone", loses.iloc[0])]
    zones = [str(z) for z in data["zones"]]
    V = np.asarray(data["V"], float).reshape(-1, 24, len(zones))
    days = np.arange(141, len(V))
    blend = np.load(res / "extra" / "floor" / "volume" / "blend_test.npy")
    yest = np.load(res / "test" / "volume" / "snaive24.npy")
    rows = []
    for role, r in pick:
        j = zones.index(str(int(r.zone)))
        for k, day in enumerate(days):
            rows.append(dict(role=role, zone=int(r.zone), skill_percent=100 * float(r.skill),
                             date=str((pd.Timestamp("2022-09-01") + pd.Timedelta(days=int(day))).date()),
                             observed_kwh=float(V[day, :, j].sum()), blend_kwh=float(blend[k, :, j].sum()),
                             yesterday_kwh=float(yest[k, :, j].sum()),
                             label="final test; rule: median winner, median loser, worst loser"))
    t = pd.DataFrame(rows)
    t.to_csv(res / "figure_representative_zones.csv", index=False)
    return t


def forecasts(res, out, data):
    """Figure 4: city-level forecasts (A-B) and representative zones (C-E), final window."""
    fc = _read(res, "figure_forecasts")
    smp = sample_table(res, data) if data is not None else _read(res, "figure_representative_zones")
    mp = _new()
    for lab, t, ylab in [("A", "volume", "Energy (million kWh/day)"), ("B", "occ", "Mean busy piles")]:
        ax = _panel(mp, lab, 70, 36, pad_left=14, pad_top=8, margin_bottom=12)
        d = fc[fc.target == t]
        x = pd.to_datetime(d.date)
        ax.axvspan(pd.Timestamp("2023-01-21"), pd.Timestamp("2023-01-28"), color=OI["yellow"], alpha=0.35, lw=0)
        ax.plot(x, d.observed, color=OI["black"], lw=1.2, label="Observed")
        ax.plot(x, d.training_floor_blend, color=OI["blue"], lw=1.2, label="Blend")
        ax.plot(x, d.yesterday, color=OI["orange"], lw=1.1, ls=":", label="Yesterday")
        ax.set_ylabel(ylab)
        ax.set_title(TN[t] + ", city mean")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        _dates(ax, "%d %b")
        if lab == "A":
            ax.legend(frameon=False, fontsize=FS - 0.5, ncol=1, loc="lower right")
    mp.newline()
    for lab, role in zip("CDE", ["Median winning zone", "Median losing zone", "Worst losing zone"]):
        ax = _panel(mp, lab, 40, 34, pad_left=14, pad_top=8, margin_bottom=12)
        d = smp[smp.role == role]
        x = pd.to_datetime(d.date)
        ax.axvspan(pd.Timestamp("2023-01-21"), pd.Timestamp("2023-01-28"), color=OI["yellow"], alpha=0.35, lw=0)
        ax.plot(x, d.observed_kwh, color=OI["black"], lw=1.1)
        ax.plot(x, d.blend_kwh, color=OI["blue"], lw=1.1)
        ax.plot(x, d.yesterday_kwh, color=OI["orange"], lw=1.0, ls=":")
        ax.set_title(f"{role} ({int(d.zone.iloc[0])}; skill {str(round(d.skill_percent.iloc[0])).replace("-", "−")}%)", fontsize=FS)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        _dates(ax, "%d %b")
        if lab == "C":
            ax.set_ylabel("Zone energy (kWh/day)")
    return _save(mp, out, "merged_forecasts")


# --------------------------------------------------------------------------- paired evidence
def evidence(res, out):
    """Figure 5: paired validation differences."""
    pi = _read(res, "figure_paired_intervals")
    mp = _new()
    # runner-up = best single model on validation: ratio LightGBM (energy), ratio XGBoost (occupancy).
    # The capped-power series exists for energy only, so that row stays empty for occupancy;
    # both panels use the same rows so the labels on the left apply to both.
    short = {"Blend − yesterday": "Blend − yesterday",
             "Blend − Ridge": "Blend − Ridge",
             "Blend − runner-up": "Blend − best single model",
             "Training floor − original": "Training floor − original floor",
             "With − without neighbours": "With − without neighbours",
             "Capped energy: blend − yesterday": "Blend − yesterday, 11 kW-capped"}
    comps = list(short)
    y = np.arange(len(comps))[::-1]
    for lab, t, first in [("A", "volume", True), ("B", "occ", False)]:
        ax = _panel(mp, lab, 52 if first else 52, 48, pad_left=70 if first else 12, pad_top=8, margin_bottom=22)
        d = pi[pi.target == t]
        ax.axvline(0, color=OI["black"], lw=0.7)
        for lv, off, c in [("zone_hour", 0.15, OI["blue"]), ("city_hour", -0.15, OI["orange"])]:
            for yi, comp in zip(y, comps):
                r = d[(d.comparison == comp) & (d.level == lv)]
                if len(r):
                    r = r.iloc[0]
                    ax.plot([r.lower, r.upper], [yi + off] * 2, color=c, lw=1.4)
                    ax.plot(r.difference, yi + off, "o", color=c, ms=3.2, label=lv.replace("_", "-") if yi == y[0] else None)
        ax.set_ylim(-0.7, len(comps) - 0.3)
        ax.set_yticks(y, [short[c] for c in comps] if first else [""] * len(comps))
        ax.set_xlabel("% of yesterday's MAE")
        ax.set_title(TN[t])
        if t == "occ":
            ax.text(0.02, y[-1], "not available", fontsize=FS - 1, color=OI["grey"], va="center", transform=ax.get_yaxis_transform())
        if first:
            ax.legend(frameon=False, fontsize=FS - 0.5, loc="lower left", bbox_to_anchor=(0, 0.08))
    return _save(mp, out, "merged_evidence")


# --------------------------------------------------------------------------- when the gain accrues
def timing(res, out):
    """Figure 6: cumulative paired loss (A-D) and blend-minus-yesterday error by hour and week (E-F)."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cum, hm = _read(res, "figure_cumulative_loss"), _read(res, "figure_error_heatmap")
    mp = _new()
    for lab, t, lv in [("A", "volume", "Zone-hour"), ("B", "volume", "City-hour"), ("C", "occ", "Zone-hour"), ("D", "occ", "City-hour")]:
        ax = _panel(mp, lab, 30, 30, pad_left=16, pad_top=12, margin_bottom=12)
        d = cum[(cum.target == t) & (cum.level == lv)]
        ax.plot(pd.to_datetime(d.date), d.cumulative_difference, color=OI["blue"] if lv == "Zone-hour" else OI["orange"], lw=1.2)
        ax.axhline(0, color=OI["black"], lw=0.6)
        ax.axvline(FINAL_START, color=OI["grey"], ls=":", lw=0.9)
        unit = {("volume", "Zone-hour"): "kWh", ("volume", "City-hour"): "kWh", ("occ", "Zone-hour"): "ratio units", ("occ", "City-hour"): "piles"}[(t, lv)]
        ax.set_title(f"{TN[t]}, {lv.lower()}\n({unit})", fontsize=FS)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        _dates(ax, "%b")
        if lab in "AC":
            ax.set_ylabel("Cumulative MAE diff.")
    mp.newline()
    cmap = LinearSegmentedColormap.from_list("bo", [OI["blue"], "#FFFFFF", OI["orange"]])
    for lab, t in [("E", "volume"), ("F", "occ")]:
        ax = _panel(mp, lab, 62, 38, pad_left=14, pad_top=8, margin_bottom=14, margin_right=24)
        d = hm[hm.target == t]
        ds = np.array(sorted(d.day.unique()))
        q = d.pivot(index="day", columns="hour", values="mae_difference").loc[ds].to_numpy()
        first = int(np.where(ds >= 141)[0][0])
        edges = sorted(set([0] + list(range(first, 0, -7))))
        edges += list(range(first + 7, len(ds), 7)) + [len(ds)]
        bins = list(zip(edges[:-1], edges[1:]))
        weekly = np.stack([q[a:b].mean(0) for a, b in bins], axis=1)
        lim = float(np.abs(weekly).max())
        im = ax.imshow(weekly, aspect="auto", origin="lower", cmap=cmap,
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), interpolation="nearest")
        ticks = list(range(0, len(bins), 4))
        ax.set_xticks(ticks, [(pd.Timestamp("2022-09-01") + pd.Timedelta(days=int(ds[bins[i][0]]))).strftime("%d %b") for i in ticks])
        ax.set_yticks([0, 6, 12, 18, 23])
        ax.set_ylabel("Target hour")
        ax.set_title(TN[t], fontsize=FS)
        ax.axvline([i for i, (a, b) in enumerate(bins) if a == first][0] - 0.5, color=OI["black"], ls="--", lw=1)
        cax = ax.inset_axes([1.03, 0, 0.04, 1])
        cb = mp.fig.colorbar(im, cax=cax)
        cb.set_ticks([-lim, 0, lim])
        cb.set_ticklabels([f"{-lim:.0f}" if t == "volume" else f"{-lim:.1f}", "0", f"{lim:.0f}" if t == "volume" else f"{lim:.1f}"])
        cb.ax.tick_params(labelsize=FS - 1)
        cb.outline.set_visible(False)
        cax.set_title("kWh" if t == "volume" else "pp", fontsize=FS - 1)
    return _save(mp, out, "merged_timing")


# --------------------------------------------------------------------------- heterogeneity
W_PRINT = 160  # mm; Figures 7, 8 and 10 (heterogeneity, investigation, commitment) are drawn at the printed text width


def _cnew():
    cns.settings.axes_titlelocation = "left"
    return cns.multipanel(max_width=W_PRINT, unit="mm")


def _cpanel(mp, lab, w, h, **kw):
    kw.setdefault("margin_right", 3)
    kw.setdefault("margin_bottom", 3)
    return mp.panel(lab, width=w, height=h, **kw)


def _cfit(mp):
    """Set tick / label sizes to the printed size (the figure is inserted at 100 %)."""
    for ax in mp.fig.axes:
        ax.tick_params(labelsize=FS, length=2, pad=1.5)
        ax.xaxis.label.set_size(FS + 0.3)
        ax.yaxis.label.set_size(FS + 0.3)


def _csave(mp, out, name):
    _cfit(mp)
    cns.savefig(str(out / (name + ".png")), fig=mp.fig)
    cns.savefig(str(out / (name + ".pdf")), fig=mp.fig)
    return name


def heterogeneity(res, out):
    zh = _read(res, "figure_zone_heterogeneity")
    profiles = _read(res, "cluster_profiles"); profiles.columns = profiles.columns.map(str)
    pn = _profile_names(profiles)
    mp = _cnew()
    H = 36
    for lab, t, w, padl in [("A", "volume", 28, 9), ("B", "occ", 28, 8)]:
        ax = _cpanel(mp, lab, w, H, pad_left=padl, pad_top=7, margin_bottom=14)
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
        ax = _cpanel(mp, lab, w, H, pad_left=padl, pad_top=7, margin_bottom=14, margin_right=0 if last else 3)
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
    lax = _cpanel(mp, " ", 150, 4, pad_left=2, pad_top=0, margin_bottom=0, margin_right=0)
    lax.axis("off")
    hs = [Line2D([0], [0], marker=mk, ls="", color=c, markerfacecolor=c if filled else "white", ms=3.4,
                 label=f"{TN[t]}, {'validation' if s_ == 'val' else 'final'}") for t, s_, c, mk, filled in series]
    lax.legend(handles=hs, loc="center", ncol=4, frameon=False, fontsize=FS - 0.3, handletextpad=0.3, columnspacing=1.6, borderpad=0)
    return _csave(mp, out, "merged_heterogeneity")


def investigation(res, out):
    tests = _read(res, "zone_failure_tests"); conc = _read(res, "zone_failure_concentration")
    traj = _read(res, "zone_failure_trajectory"); abl = _read(res, "ablation_steps"); feat = _read(res, "figure_feature_gain")
    names = {"zero_share": "Zero-hour share", "festival_ratio": "Festival ratio", "max_daily_jump_final": "Largest daily jump", "test_vs_last30": "Level vs last 30 d"}
    mp = _cnew()
    H1, H2 = 30, 25
    ax = _cpanel(mp, "A", 30, H1, pad_left=32, pad_top=6, margin_bottom=25)
    labs, est, lo, hi, cols = [], [], [], [], []
    for t, c in [("volume", OI["blue"]), ("occ", OI["orange"])]:
        for k in names:
            r = tests[(tests.target == t) & (tests.covariate == k)].iloc[0]
            labs.append(f"{names[k]} ({TN[t][0]})"); est.append(r.auc); lo.append(r.auc_lower); hi.append(r.auc_upper); cols.append(c)
    _hforest(ax, labs, est, lo, hi, cols, ref=0.5)
    ax.set_xlabel("AUC, losing vs other zones")
    ax.set_title("Who loses", fontsize=FS)
    ax = _cpanel(mp, "B", 25, H1, pad_left=23, pad_top=6, margin_bottom=25)
    a = abl[(abl.split == "test") & (abl.target == "volume")]
    a = a[a.step.str.startswith("Total") == False]
    labs = ["Add neighbours", "Ratio target", "Training floor", "XGBoost blend"]
    _hforest(ax, labs, a.percent, a.percent_lower, a.percent_upper, [OI["blue"]] * len(a))
    ax.set_xlabel("Energy MAE change (%)")
    ax.set_title("Design steps", fontsize=FS)
    ax = _cpanel(mp, "C", 24, H1, pad_left=10, pad_top=6, margin_bottom=25, margin_right=0)
    for t, c in [("volume", OI["blue"]), ("occ", OI["orange"])]:
        d = conc[conc.target == t]
        ax.plot(range(len(d)), 100 * d.excess_share, "o-", color=c, ms=2.8, lw=1.0, label=TN[t] if t == "volume" else "Occup.")
    ax.set_xticks(range(4), ["1", "5", "10", "51"])
    ax.set_xlabel("Worst losing zones")
    ax.set_ylabel("Share of excess error (%)")
    ax.legend(frameon=False, fontsize=FS - 0.5, loc="lower right", handlelength=1.2, bbox_to_anchor=(1.04, -0.02), labelspacing=0.3, handletextpad=0.4)
    ax.set_title("Concentration", fontsize=FS)
    mp.newline()
    ax = _cpanel(mp, "D", 85, H2, pad_left=10, pad_top=8, margin_bottom=12)
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
    ax = _cpanel(mp, "E", 46, H2, pad_left=10, pad_top=8, margin_bottom=12, margin_right=0)
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
    return _csave(mp, out, "merged_investigation")


def holidays(res, out):
    """Figure 9: three holiday regimes, both targets (grid of identical panels)."""
    h = _read(res, "figure_holiday_comparison")
    mp = _new()
    sty = {"Observed": (OI["black"], "-"), "Training-floor blend": (OI["blue"], "-"), "Yesterday": (OI["orange"], ":")}
    events = ["National Day", "New Year", "Spring Festival*"]
    labs = iter("ABCDEF")
    for row, t in enumerate(["volume", "occ"]):
        for col, ev in enumerate(events):
            ax = _panel(mp, next(labs), 48, 30, pad_left=16 if col == 0 else 8, pad_top=8, margin_bottom=12)
            d = h[(h.event == ev) & (h.target == t)]
            for m, (c, ls) in sty.items():
                r = d[d.model == m]
                ax.plot(r.relative_day, r.value / (1000 if t == "volume" else 1), color=c, ls=ls, lw=1.2, label=m)
            ax.set_title(f"{ev.replace('*', ' (final test)')}" if row == 0 else "", fontsize=FS)
            if col == 0:
                ax.set_ylabel("Energy (million kWh/day)" if t == "volume" else "Busy piles (mean)")
            if row == 1:
                ax.set_xlabel("Day relative to first holiday day")
            if row == 0 and col == 2:
                ax.legend(frameon=False, fontsize=FS - 1.5, loc="upper left")
        if row == 0:
            mp.newline()
    return _save(mp, out, "merged_holidays")


# --------------------------------------------------------------------------- commitment
def commitment(res, out):
    fp = _read(res / "decisions", "full_policy_costs")
    fp = fp[(fp.split == "val") & (fp.subset == "common_folds_2_to_4_81")]
    ct = _read(res / "decisions", "raw_vs_scaled_contrasts")
    ct = ct[(ct.split == "val") & (ct.subset == "common_folds_2_to_4_81") & (ct.level == "zone_hour")
            & (ct.cost_ratio == 4) & (ct.block_days == 7)]
    im = _read(res, "interval_metrics")
    mp = _cnew()
    sty = {"seasonal_naive_scaled": ("Yesterday × k", OI["orange"], "--"), "ML_point": ("ML point", OI["sky"], ":"),
           "ML_point_scaled": ("ML point × k", OI["blue"], "-."), "ML_raw_quantile": ("Raw quantile", OI["purple"], "-"),
           "ML_calibrated_quantile": ("Calibrated quantile", OI["grey"], "--")}
    H = 40
    for lab, lv, w, padl in [("A", "zone_hour", 27, 9), ("B", "city_hour", 25, 8)]:
        ax = _cpanel(mp, lab, w, H, pad_left=padl, pad_top=7, margin_bottom=19)
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
    ax = _cpanel(mp, "C", 25, H, pad_left=9, pad_top=7, margin_bottom=19)
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
    ax = _cpanel(mp, "D", 27, H, pad_left=9, pad_top=7, margin_bottom=19, margin_right=0)
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
    lax = _cpanel(mp, " ", 150, 4, pad_left=2, pad_top=0, margin_bottom=0, margin_right=0)
    lax.axis("off")
    hs = [Line2D([0], [0], color=c, ls=ls, marker="o", ms=2.2, lw=1.0, label=n) for n, c, ls in sty.values()]
    hs += [Line2D([0], [0], marker="o", ls="", color=OI["black"], markerfacecolor="white", ms=3, label="Raw (D)"),
           Line2D([0], [0], marker="o", ls="", color=OI["black"], markerfacecolor=OI["black"], ms=3, label="Calibrated (D)")]
    lax.legend(handles=hs, loc="center", ncol=7, frameon=False, fontsize=FS - 0.5, handlelength=1.6, columnspacing=0.8, borderpad=0)
    return _csave(mp, out, "merged_commitment")


def merged(out, results, data=None):
    out, res = Path(out) / "figures", Path(results)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = None if data is None else data.get("data_path")
    names = []
    if data_dir is not None:
        names.append(overview(res, out, data_dir))
    names += [accuracy(res, out), forecasts(res, out, data), evidence(res, out), timing(res, out),
              heterogeneity(res, out), investigation(res, out), holidays(res, out), commitment(res, out)]
    return names
