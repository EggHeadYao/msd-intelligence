#!/usr/bin/env python3
"""Generate schematic Learned Ranker (R) figures for the MERLIN presentation.

These diagrams encode the implemented pipeline but deliberately contain no
experimental result values.  They are exported as PDF and SVG so the slides
remain reproducible and the vector artwork can be inspected independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "background": "#E9F0EF",
    "teal": "#236B65",
    "blue": "#3E7198",
    "purple": "#76689A",
    "gold": "#C98B38",
    "ink": "#173A38",
    "grey": "#7B8584",
    "light": "#F4F7F6",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "savefig.facecolor": COLORS["background"],
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def canvas(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    edge: str,
    face: str = "white",
    fontsize: float = 10.5,
    weight: str = "normal",
    text_color: str | None = None,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color or COLORS["ink"],
        linespacing=1.25,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.7,
            color=color or COLORS["teal"],
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


def save(fig, output_dir: Path, stem: str) -> None:
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )


def candidate_pipeline(output_dir: Path) -> None:
    fig, ax = canvas((11.4, 3.0))
    sources = (
        ("C1 Audio", COLORS["blue"], "#EAF0F5"),
        ("C2 Graph", COLORS["teal"], "#E7F0EF"),
        ("BFS", COLORS["purple"], "#F0EDF5"),
        ("Tag", COLORS["gold"], "#F8F0E4"),
    )
    ys = (0.76, 0.54, 0.32, 0.10)
    for (label, color, face), y in zip(sources, ys, strict=True):
        box(ax, 0.02, y, 0.23, 0.14, label, edge=color, face=face, weight="bold")
        arrow(ax, (0.25, y + 0.07), (0.34, 0.50), color=color)

    box(
        ax,
        0.34,
        0.27,
        0.28,
        0.46,
        "Filter\n+\nMerge",
        edge=COLORS["ink"],
        face=COLORS["light"],
        fontsize=13,
        weight="bold",
    )
    arrow(ax, (0.62, 0.50), (0.69, 0.50))
    box(
        ax,
        0.69,
        0.34,
        0.17,
        0.32,
        "≤ 1K\nCandidates",
        edge=COLORS["teal"],
        face="#E7F0EF",
        fontsize=12,
        weight="bold",
    )
    arrow(ax, (0.86, 0.50), (0.91, 0.50))
    box(
        ax,
        0.91,
        0.37,
        0.075,
        0.26,
        "R",
        edge=COLORS["gold"],
        face="#F8F0E4",
        fontsize=11,
        weight="bold",
    )
    save(fig, output_dir, "c3_candidate_pipeline")


def pair_construction(output_dir: Path) -> None:
    fig, ax = canvas((11.4, 2.7))
    box(
        ax,
        0.04,
        0.32,
        0.17,
        0.36,
        "Song-safe\nSplit",
        edge=COLORS["ink"],
        face=COLORS["light"],
        fontsize=12,
        weight="bold",
    )
    box(
        ax,
        0.29,
        0.32,
        0.18,
        0.36,
        "Training\nPairs",
        edge=COLORS["blue"],
        face="#EAF0F5",
        fontsize=12,
        weight="bold",
    )
    box(
        ax,
        0.55,
        0.32,
        0.16,
        0.36,
        "13 Pair\nFeatures",
        edge=COLORS["purple"],
        face="#F0EDF5",
        fontsize=12,
        weight="bold",
    )
    box(
        ax,
        0.79,
        0.32,
        0.17,
        0.36,
        "Logistic\nRegression",
        edge=COLORS["gold"],
        face="#F8F0E4",
        fontsize=12,
        weight="bold",
    )
    arrow(ax, (0.21, 0.50), (0.29, 0.50))
    arrow(ax, (0.47, 0.50), (0.55, 0.50))
    arrow(ax, (0.71, 0.50), (0.79, 0.50), color=COLORS["gold"])
    save(fig, output_dir, "c3_pair_construction")


def ranker_architecture(output_dir: Path) -> None:
    fig, ax = canvas((11.4, 2.8))
    box(
        ax,
        0.02,
        0.35,
        0.12,
        0.30,
        "Canonical\ncandidate pool",
        edge=COLORS["ink"],
        face=COLORS["light"],
        weight="bold",
    )
    box(
        ax,
        0.19,
        0.29,
        0.24,
        0.42,
        "13 pair features\n\n6 evidence values\n5 availability masks\n2 interactions",
        edge=COLORS["teal"],
        face="#E7F0EF",
        weight="bold",
    )
    box(
        ax,
        0.49,
        0.32,
        0.18,
        0.36,
        "Set-A transform\nmedian fill\n+ external scaling",
        edge=COLORS["blue"],
        face="#EAF0F5",
        weight="bold",
    )
    box(
        ax,
        0.73,
        0.32,
        0.13,
        0.36,
        "Weighted LR\nraw margin",
        edge=COLORS["gold"],
        face="#F8F0E4",
        weight="bold",
    )
    box(
        ax,
        0.91,
        0.35,
        0.075,
        0.30,
        "Ranked\nTop-20",
        edge=COLORS["teal"],
        face="#E7F0EF",
        weight="bold",
    )
    arrow(ax, (0.14, 0.50), (0.19, 0.50))
    arrow(ax, (0.43, 0.50), (0.49, 0.50))
    arrow(ax, (0.67, 0.50), (0.73, 0.50))
    arrow(ax, (0.86, 0.50), (0.91, 0.50), color=COLORS["gold"])
    ax.text(
        0.31,
        0.13,
        "Audio  •  Graph  •  BFS  •  Tag  •  Release  •  Year",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["grey"],
    )
    save(fig, output_dir, "c3_ranker_architecture")


def selection_policy(output_dir: Path) -> None:
    fig, ax = canvas((11.4, 2.5))
    box(
        ax,
        0.04,
        0.48,
        0.14,
        0.32,
        "Set A\nTrain",
        edge=COLORS["blue"],
        face="#EAF0F5",
        fontsize=11.5,
        weight="bold",
    )
    box(
        ax,
        0.24,
        0.48,
        0.14,
        0.32,
        "Set-B\nTune",
        edge=COLORS["purple"],
        face="#F0EDF5",
        fontsize=11.5,
        weight="bold",
    )
    box(
        ax,
        0.44,
        0.48,
        0.14,
        0.32,
        "Set-B\nConfirm",
        edge=COLORS["purple"],
        face="#F0EDF5",
        fontsize=11.5,
        weight="bold",
    )
    box(
        ax,
        0.64,
        0.48,
        0.14,
        0.32,
        "Full\nRetrain",
        edge=COLORS["teal"],
        face="#E7F0EF",
        fontsize=11.5,
        weight="bold",
    )
    box(
        ax,
        0.84,
        0.48,
        0.14,
        0.32,
        "Set-C\nDev",
        edge=COLORS["gold"],
        face="#F8F0E4",
        fontsize=11.5,
        weight="bold",
    )
    arrow(ax, (0.18, 0.64), (0.24, 0.64))
    arrow(ax, (0.38, 0.64), (0.44, 0.64))
    arrow(ax, (0.58, 0.64), (0.64, 0.64))
    arrow(ax, (0.78, 0.64), (0.84, 0.64), color=COLORS["gold"])
    ax.text(
        0.50,
        0.22,
        "Same queries  •  same pool  •  same budget",
        ha="center",
        va="center",
        fontsize=11.5,
        color=COLORS["teal"],
        fontweight="bold",
    )
    save(fig, output_dir, "c3_selection_policy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    candidate_pipeline(args.output_dir)
    pair_construction(args.output_dir)
    ranker_architecture(args.output_dir)
    selection_policy(args.output_dir)


if __name__ == "__main__":
    main()
