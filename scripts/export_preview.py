"""Render the project-page figure from the existing synthetic-data starter.

Run from any directory with the repository's requirements installed:
    python scripts/export_preview.py
"""

from pathlib import Path
import runpy
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # Execute the original data generator without opening its interactive plot.
    with patch.object(plt, "show"):
        source = runpy.run_path(str(root / "01_exact_dmd_numpy.py"))
    plt.close("all")

    field = source["X_data"]
    x = source["x"]
    t = source["t"]
    if field.shape != (len(x), len(t)) or not np.isfinite(field).all():
        raise ValueError("Expected a finite space-by-time snapshot array.")

    background = "#f2f1eb"
    foreground = "#35434c"
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": foreground,
        "axes.labelcolor": foreground,
        "xtick.color": foreground,
        "ytick.color": foreground,
        "axes.edgecolor": "#c4c8c5",
        "axes.linewidth": 0.6,
        "figure.facecolor": background,
        "axes.facecolor": background,
        "savefig.facecolor": background,
    }):
        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        limit = float(np.abs(field).max())
        plot = ax.imshow(
            field,
            aspect="auto",
            origin="lower",
            extent=[float(t[0]), float(t[-1]), float(x[0]), float(x[-1])],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.set_xlabel("Time, t", labelpad=10)
        ax.set_ylabel("Spatial coordinate, x", labelpad=8)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.tick_params(length=3, width=0.6, labelsize=9, pad=5)
        colorbar = fig.colorbar(plot, ax=ax, fraction=0.035, pad=0.035)
        colorbar.set_label("u(x, t)", labelpad=9)
        colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        colorbar.ax.tick_params(length=2, width=0.6, labelsize=9)
        colorbar.outline.set_visible(False)

        destination = root / "docs" / "assets" / "synthetic-field.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180)

        # A separate compact render keeps chart labels readable on phones.
        fig.set_size_inches(4.3, 3.8)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
        ax.tick_params(labelsize=10)
        ax.set_yticks([-10, 0, 10])
        colorbar.set_ticks([-1, 0, 1])
        colorbar.ax.tick_params(labelsize=10)
        colorbar.ax.yaxis.label.set_fontsize(11)
        mobile_destination = destination.with_stem("synthetic-field-mobile")
        fig.savefig(mobile_destination, dpi=180)
        plt.close(fig)

    print(f"Saved {destination}")
    print(f"Saved {mobile_destination}")
    print(f"Snapshot array: {field.shape[0]} spatial points x {field.shape[1]} snapshots")


if __name__ == "__main__":
    main()
