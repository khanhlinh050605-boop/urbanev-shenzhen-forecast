"""Plot style."""
import matplotlib.pyplot as plt


STYLE = {
    "width_cm": 17,
    "dpi": 220,
    "font": "Times New Roman",
    "font_size": 9,
    "ink": "#223047",
    "muted": "#697386",
    "grid": "#E5E9EF",
    "observed": "#223047",
    "yesterday": "#D69424",
    "blend": "#2463A0",
    "floor": "#007F7B",
    "raw": "#D69424",
    "calibrated": "#8056A1",
    "city": "#D69424",
    "zone": "#2463A0",
    "bad": "#C76C3A",
    "holiday": "#FFF1C7",
    "swatch_edge": "#D8DEE7",
    "test": "#EEE9F4",
    "cluster0": "#2463A0",
    "cluster1": "#D69424",
}


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                STYLE["font"],
                "Liberation Serif",
                "TeX Gyre Termes",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "font.size": STYLE["font_size"],
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#B1BBC7",
            "axes.labelcolor": STYLE["ink"],
            "text.color": STYLE["ink"],
            "xtick.color": STYLE["muted"],
            "ytick.color": STYLE["muted"],
            "axes.axisbelow": True,
            "grid.color": STYLE["grid"],
            "grid.linewidth": 0.65,
            "lines.linewidth": 1.5,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "axes.formatter.useoffset": False,
            "axes.formatter.limits": (-12, 12),
        }
    )
