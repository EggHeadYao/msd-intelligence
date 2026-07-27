from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_common import BACKGROUND, EXPERIMENT_DIR, REPOSITORY_DIR
from plot_common import bold_ticks, configure_axis, configure_plots, load_rows, save_figure

MODELS = (
    ("ridge-t90", "Full batch", "#173F5F"),
    ("ridge-t90-mb25", "Mini-batch 25%", "#D1495B"),
    ("ridge-t90-mb10", "Mini-batch 10%", "#2A9D8F"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Ridge batching convergence.")
    parser.add_argument("--input", type=Path, default=EXPERIMENT_DIR / "convergence.csv")
    parser.add_argument("--output", type=Path, default=REPOSITORY_DIR / "slides" / "img")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    configure_plots()
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), facecolor=BACKGROUND)
    panels = (
        ("cumulative_seconds", "Training-loop time (seconds)", False),
        ("effective_data_passes", "Equivalent full-data passes (log scale)", True),
    )
    for axis, (field, label, logarithmic) in zip(axes, panels):
        configure_axis(axis)
        for model_id, model_label, color in MODELS:
            selected = [row for row in rows if row["model_id"] == model_id]
            axis.plot(
                [float(row[field]) for row in selected],
                [float(row["validation_mae"]) for row in selected],
                color=color,
                label=model_label,
                linewidth=3.0,
                marker="o",
                markersize=5.2,
            )
        axis.axhline(6.461948, color="#6B7280", linestyle="--", linewidth=1.4)
        axis.set_xlabel(label)
        axis.set_ylabel("Validation MAE (years)")
        if logarithmic:
            axis.set_xscale("log")
            axis.text(
                0.03,
                6.467,
                "Target: full-batch final MAE",
                color="#6B7280",
                fontsize=11,
                fontweight="bold",
                ha="left",
                transform=axis.get_yaxis_transform(),
            )
        bold_ticks(axis)
    axes[0].legend(frameon=False, loc="upper right", prop={"size": 12, "weight": "bold"})
    figure.suptitle(
        "Mini-batches reach the same Ridge quality with less work",
        fontsize=18,
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(figure, args.output, "ridge_batching_convergence")


if __name__ == "__main__":
    main()
