"""Figure: cluster geography."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Patch
from pyproj import CRS, Transformer

from ...config import SEED
from ..style import STYLE


def draw(ctx):
    V = ctx.V
    data = ctx.data
    footnote = ctx.footnote
    label_ticks = ctx.label_ticks
    read = ctx.read
    res = ctx.res
    save = ctx.save
    z = ctx.z
    zones = ctx.zones

    clusters = read("zone_clusters")
    profiles = read("cluster_profiles")
    if clusters.empty:
        from sklearn.cluster import KMeans

        pp = V[:30].mean(0).T
        pp /= np.maximum(pp.mean(1, keepdims=True), 1e-12)
        group = KMeans(2, n_init=20, random_state=SEED).fit_predict(pp)
        clusters = pd.DataFrame(
            dict(
                zone=zones,
                cluster=group,
                training_mean_kwh=V[:30].mean((0, 1)),
                training_volume_share=V[:30].sum((0, 1)) / V[:30].sum(),
            )
        )
        profiles = pd.DataFrame(pp).assign(zone=zones, cluster=group)
        clusters.to_csv(res / "zone_clusters.csv", index=False)
        profiles.to_csv(res / "cluster_profiles.csv", index=False)
    profiles.columns = profiles.columns.map(str)
    clusters.zone = clusters.zone.astype(str)
    cluster_map = clusters.set_index("zone").cluster.to_dict()
    profile_names = {}
    for group, f in profiles.groupby("cluster"):
        pp = f[[str(h) for h in range(24)]].mean().to_numpy()
        peak = int(pp.argmax())
        profile_names[int(group)] = "Night-peaked" if peak < 8 or peak >= 22 else "Day-peaked"
    shapepath = (
        Path(data.get("data_dir", data.get("data_path", data.get("path", "data"))))
        / "SZ_districts"
        / "SZ_districts.shp"
    )
    if not shapepath.exists():
        raise FileNotFoundError(shapepath)
    if shapepath.exists():
        reader = shapefile.Reader(str(shapepath))
        crs = CRS.from_wkt(shapepath.with_suffix(".prj").read_text())
        project = Transformer.from_crs(crs, 4326, always_xy=True)
        fig = plt.figure(figsize=(STYLE["width_cm"] / 2.54, 8.4 / 2.54))
        gs = fig.add_gridspec(
            1, 2, width_ratios=[1.35, 1], left=0.09, right=0.975, top=0.83, bottom=0.12, wspace=0.3
        )
        a1 = fig.add_subplot(gs[0, 0])
        a2 = fig.add_subplot(gs[0, 1])
        patches = []
        colours = []
        chartrows = []
        matched = set()
        for item in reader.iterShapeRecords():
            raw = float(item.record["TAZID"])
            assert raw.is_integer()
            zone = str(int(raw))
            group = cluster_map.get(zone)
            if group is not None:
                matched.add(zone)
            parts = list(item.shape.parts) + [len(item.shape.points)]
            for a, b in zip(parts[:-1], parts[1:]):
                xy = np.asarray(item.shape.points[a:b])
                lon, lat = project.transform(xy[:, 0], xy[:, 1])
                patches.append(Polygon(np.column_stack([lon, lat])))
                colours.append(STYLE["cluster" + str(group)] if group is not None else "#E4E8ED")
            chartrows.append(
                dict(
                    panel="map",
                    zone=zone,
                    cluster=group,
                    matched=group is not None,
                    label="Initial training only",
                )
            )
        a1.add_collection(
            PatchCollection(patches, facecolors=colours, edgecolors="white", linewidths=0.2)
        )
        a1.autoscale_view()
        a1.set_anchor("N")
        a1.set_aspect(1 / np.cos(np.deg2rad(22.6)))
        a1.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)", title="Zone clusters, Shenzhen")
        a1.ticklabel_format(useOffset=False)
        handles = []
        for group, f in profiles.groupby("cluster"):
            group = int(group)
            pp = f[[str(h) for h in range(24)]].mean().to_numpy()
            colour = STYLE["cluster" + str(group)]
            name = f"{profile_names[group]} ({len(f)} zones)"
            a2.plot(range(24), pp, color=colour, lw=1.7)
            a2.fill_between(range(24), 1, pp, color=colour, alpha=0.10, lw=0)
            handles.append(Patch(facecolor=colour, label=name))
            chartrows.extend(
                dict(
                    panel="profile",
                    cluster=group,
                    hour=h,
                    normalized_profile=v,
                    training_zones=len(f),
                    label="Initial training only",
                )
                for h, v in enumerate(pp)
            )
        handles.append(Patch(facecolor="#E4E8ED", label=f"Outside the {z}-zone sample"))
        a2.axhline(1, color=STYLE["muted"], ls=":", lw=1.3)
        a2.set(
            title="Average training profile", xlabel="Hour of day", ylabel="Relative to zone mean"
        )
        a2.set_xticks([0, 6, 12, 18, 23])
        label_ticks(a2)
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            frameon=False,
            ncol=3,
            fontsize=8.2,
        )
        footnote(
            fig,
            f"Exact zone-ID match: {len(matched)} of {z}. Clusters use only the first 30 training days; no activity is inferred for grey polygons.",
        )
        save("cluster_geography", fig, chartrows)
        yield "cluster_geography"
    ctx.cluster_map = cluster_map
    ctx.profile_names = profile_names
