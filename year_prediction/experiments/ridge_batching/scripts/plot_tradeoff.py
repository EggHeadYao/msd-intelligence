from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_common import BACKGROUND, EXPERIMENT_DIR, REPOSITORY_DIR
from plot_common import bold_ticks, configure_axis, configure_plots, load_rows, save_figure
from tradeoff_panels import plot_cost, plot_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Ridge batching trade-offs.")
    parser.add_argument("--output", type=Path, default=REPOSITORY_DIR / "slides" / "img")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    efficiency = load_rows(EXPERIMENT_DIR / "efficiency.csv")
    quality = load_rows(EXPERIMENT_DIR / "quality.csv")
    configure_plots()
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), facecolor=BACKGROUND)
    for axis in axes:
        configure_axis(axis)
    plot_cost(axes[0], efficiency)
    plot_quality(axes[1], quality)
    for axis in axes:
        bold_ticks(axis)
    figure.suptitle(
        "Mini-batching cuts work without changing test quality",
        fontsize=18,
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(figure, args.output, "ridge_batching_tradeoff")


if __name__ == "__main__":
    main()
