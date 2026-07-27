from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


LABELS = ("Full batch", "Mini-batch 25%", "Mini-batch 10%")
WIDTH = 0.34


def plot_cost(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    positions = np.arange(3)
    full_time = float(rows[0]["total_seconds"])
    times = [100 * float(row["total_seconds"]) / full_time for row in rows]
    passes = [float(row["effective_data_passes"]) for row in rows]
    time_bars = axis.bar(
        positions - WIDTH / 2, times, WIDTH, color="#D1495B", label="Total time"
    )
    pass_bars = axis.bar(
        positions + WIDTH / 2,
        passes,
        WIDTH,
        color="#2A9D8F",
        label="Data passes",
    )
    axis.bar_label(
        time_bars,
        labels=[f"{value:.1f}%" for value in times],
        padding=3,
        fontweight="bold",
    )
    axis.bar_label(
        pass_bars,
        labels=["", *(f"{value:.1f}%" for value in passes[1:])],
        padding=3,
        fontweight="bold",
    )
    axis.set_title("Normalized training cost")
    axis.set_ylabel("Full batch = 100%")
    axis.set_xticks(positions, LABELS)
    axis.legend(frameon=False, prop={"size": 12, "weight": "bold"})


def plot_quality(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    baseline_mae = float(rows[0]["test_mae"])
    baseline_rmse = float(rows[0]["test_rmse"])
    mae = [365.25 * (float(row["test_mae"]) - baseline_mae) for row in rows[1:]]
    rmse = [365.25 * (float(row["test_rmse"]) - baseline_rmse) for row in rows[1:]]
    positions = np.arange(2)
    mae_bars = axis.bar(
        positions - WIDTH / 2, mae, WIDTH, color="#D1495B", label="MAE"
    )
    rmse_bars = axis.bar(
        positions + WIDTH / 2, rmse, WIDTH, color="#547AA5", label="RMSE"
    )
    axis.bar_label(
        mae_bars, labels=[f"{value:+.2f} d" for value in mae], fontweight="bold"
    )
    axis.bar_label(
        rmse_bars, labels=[f"{value:+.2f} d" for value in rmse], fontweight="bold"
    )
    axis.axhline(0, color="#334155", linewidth=1.2)
    axis.set_title("Test change relative to full batch")
    axis.set_ylabel("Difference (days)")
    axis.set_xticks(positions, LABELS[1:])
    axis.legend(frameon=False, prop={"size": 12, "weight": "bold"})
