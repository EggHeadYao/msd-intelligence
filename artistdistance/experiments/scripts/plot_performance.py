from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from plot_common import BACKGROUND, EXPERIMENT_DIR, REPOSITORY_DIR
from plot_common import bold_ticks, configure_axis, configure_plots, load_rows, save_figure


ENGINES = (("spark", "Spark", "#2A9D8F"), ("mapreduce", "MapReduce", "#D1495B"))
FORMATS = (("avro", "Avro", 1.0), ("parquet", "Parquet", 0.32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot artist-distance performance.")
    parser.add_argument("--input", type=Path, default=EXPERIMENT_DIR / "summary.csv")
    parser.add_argument("--output", type=Path, default=REPOSITORY_DIR / "slides" / "img")
    return parser.parse_args()


def duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    by_key = {(row["engine"], row["format"]): row for row in rows}
    configure_plots()
    figure, axis = plt.subplots(figsize=(12.8, 5.3), facecolor=BACKGROUND)
    configure_axis(axis)

    for format_id, format_label, position in FORMATS:
        medians = {engine: float(by_key[engine, format_id]["median_wall_seconds"]) for engine, _, _ in ENGINES}
        axis.plot([medians["spark"], medians["mapreduce"]], [position, position], color="#80908E", linewidth=2.2)
        speedup = medians["mapreduce"] / medians["spark"]
        axis.text(math.sqrt(medians["spark"] * medians["mapreduce"]), position + 0.13, f"{speedup:.1f}x faster", ha="center", color="#334155")
        for engine, engine_label, color in ENGINES:
            row = by_key[engine, format_id]
            median = medians[engine]
            low = median - float(row["min_wall_seconds"])
            high = float(row["max_wall_seconds"]) - median
            axis.errorbar(median, position, xerr=[[low], [high]], fmt="o", color=color, markersize=11, capsize=6, elinewidth=2.2, label=engine_label if position == 1 else None)
            offset, alignment = (1.09, "left") if engine == "spark" else (0.92, "right")
            axis.text(median * offset, position - 0.14, duration(median), ha=alignment, color=color)

    axis.set_xscale("log")
    axis.set_xlim(60, 1800)
    axis.set_ylim(0, 1.38)
    axis.set_xticks((60, 120, 300, 600, 1200, 1800), ("1m", "2m", "5m", "10m", "20m", "30m"))
    axis.set_yticks((1.0, 0.32), ("Avro", "Parquet"))
    axis.set_xlabel("Submitted BFS wall time (log scale)")
    axis.tick_params(axis="y", length=0, pad=14)
    axis.legend(frameon=False, loc="upper right", ncol=2, prop={"size": 15, "weight": "bold"})
    bold_ticks(axis)
    figure.suptitle("Spark accelerates distributed BFS by 14x", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    save_figure(figure, args.output, "artistdistance_performance")


if __name__ == "__main__":
    main()
