from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = EXPERIMENT_DIR.parents[1]
BACKGROUND = "#E9F0EF"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )


def configure_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color="#AAB7B5", alpha=0.45, linewidth=0.8)


def bold_ticks(axis: plt.Axes) -> None:
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor=BACKGROUND)
    figure.savefig(
        output / f"{name}.png", dpi=220, bbox_inches="tight", facecolor=BACKGROUND
    )
