#!/usr/bin/env python3
"""Generate MERLIN evidence figures from formal JSON reports.

The script intentionally consumes report summaries instead of reconstructing or
inventing observations.  All figures are vector PDFs suitable for Beamer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "background": "#E9F0EF",
    "teal": "#236B65",
    "blue": "#3E7198",
    "purple": "#76689A",
    "gold": "#C98B38",
    "ink": "#173A38",
    "grey": "#7B8584",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "savefig.facecolor": COLORS["background"],
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
        }
    )


def c1_empirical_summary(report: dict, output: Path) -> None:
    distributions = report["distributions"]["pca_128"]
    effects = report["effect_size_vs_random"]["pca_128"]
    names = ["Matched random", "Same artist", "Same release"]
    keys = ["random", "same_artist", "same_release"]
    colors = [COLORS["grey"], COLORS["blue"], COLORS["teal"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.25), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    y = np.arange(len(keys))[::-1]
    for yi, key, color in zip(y, keys, colors):
        d = distributions[key]
        ax.plot([d["p05"], d["p95"]], [yi, yi], color=color, lw=2, solid_capstyle="round")
        ax.plot([d["p25"], d["p75"]], [yi, yi], color=color, lw=8, solid_capstyle="round")
        ax.scatter(d["p50"], yi, s=34, color="white", edgecolor=color, linewidth=1.4, zorder=3)
        ax.scatter(d["mean"], yi, s=48, color=color, marker="D", zorder=3)
    ax.axvline(0, color="#D7DEDD", lw=0.8, zorder=0)
    ax.set_yticks(y, names)
    ax.set_xlabel("PCA-128 cosine similarity")
    ax.set_title("Empirical pair summaries (10K each)", loc="left", fontweight="bold")
    ax.text(0.99, -0.27, "thin: 5–95%  thick: 25–75%  ○ median  ◆ mean",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=COLORS["grey"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    effect_keys = ["same_artist", "same_release"]
    effect_names = ["Same artist", "Same release"]
    effect_colors = [COLORS["blue"], COLORS["teal"]]
    ey = np.arange(2)[::-1]
    for yi, key, color in zip(ey, effect_keys, effect_colors):
        e = effects[key]
        low, high = e["bootstrap_95_ci"]
        point = e["hedges_g"]
        ax.errorbar(point, yi, xerr=[[point - low], [high - point]], fmt="o", ms=7,
                    color=color, ecolor=color, capsize=4, lw=2)
        ax.text(high + 0.035, yi, f"{point:.2f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0.9, 1.48)
    ax.set_yticks(ey, effect_names)
    ax.set_xlabel("Hedges' g vs matched random")
    ax.set_title("Effect size with 95% bootstrap CI", loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout(w_pad=2.4)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def candidate_recall(report: dict, output: Path) -> None:
    layer = report["candidate_layer"]
    strata = ["audio_dominant", "mixed", "relation_dominant"]
    labels = ["Audio-dominant", "Mixed", "Relation-dominant"]
    sources = ["audio", "graph", "bfs", "tag"]
    source_labels = ["C1", "C2", "BFS", "Tag"]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["purple"], COLORS["gold"]]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), sharey=True)
    for ax, stratum, label in zip(axes, strata, labels):
        values = [100 * layer[stratum]["single_source_recall@250"].get(source, 0.0) for source in sources]
        union = 100 * layer[stratum]["union_recall@1000"]
        bars = ax.bar(source_labels, values, color=colors, width=0.68)
        ax.axhline(union, color=COLORS["ink"], lw=1.6, ls="--")
        ax.text(0.98, union + 0.035, f"Union ≤1K: {union:.2f}%", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(label, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
        for bar, value in zip(bars, values):
            if value >= 0.08:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}",
                        ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Candidate recall (%)")
    axes[0].set_ylim(0, 2.15)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-report", type=Path, required=True)
    parser.add_argument("--set-c-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    c1_report = load_json(args.c1_report)
    set_c_report = load_json(args.set_c_report)
    c1_empirical_summary(c1_report, args.output_dir / "c1_empirical_summary.pdf")
    candidate_recall(set_c_report, args.output_dir / "candidate_recall.pdf")


if __name__ == "__main__":
    main()
