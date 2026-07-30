from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pyspark import cloudpickle

from .features import Point, point_batches

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@dataclass(frozen=True)
class Statistics:
    count: int
    loss_sum: float
    gradient_sum: np.ndarray
    intercept_gradient_sum: float

    def merge(self, other: "Statistics") -> "Statistics":
        return Statistics(
            self.count + other.count,
            self.loss_sum + other.loss_sum,
            self.gradient_sum + other.gradient_sum,
            self.intercept_gradient_sum + other.intercept_gradient_sum,
        )


def residual_terms(residuals: np.ndarray, loss: str, delta: float) -> tuple[np.ndarray, np.ndarray]:
    if loss == "squared":
        return 0.5 * residuals * residuals, residuals
    if loss != "huber" or delta <= 0.0:
        raise ValueError("loss must be squared or Huber with positive delta")
    absolute = np.abs(residuals)
    quadratic = absolute <= delta
    losses = np.where(quadratic, 0.5 * residuals * residuals, delta * (absolute - 0.5 * delta))
    derivatives = np.where(quadratic, residuals, delta * np.sign(residuals))
    return losses, derivatives


def partition_statistics(
    rows: Iterable[Point], weights: np.ndarray, intercept: float, loss: str, delta: float
) -> Iterable[Statistics]:
    count = 0
    loss_sum = 0.0
    gradient = np.zeros_like(weights)
    intercept_gradient = 0.0
    for points in point_batches(rows):
        features = np.stack([row[4] for row in points])
        labels = np.asarray([row[3] for row in points], dtype=np.float64)
        residuals = features @ weights + intercept - labels
        losses, derivatives = residual_terms(residuals, loss, delta)
        count += len(points)
        loss_sum += float(np.sum(losses, dtype=np.float64))
        gradient += features.T @ derivatives
        intercept_gradient += float(np.sum(derivatives, dtype=np.float64))
    if count:
        yield Statistics(count, loss_sum, gradient, intercept_gradient)


def distributed_statistics(points, weights, intercept, loss, delta, l2) -> tuple[float, np.ndarray, float]:
    zero = Statistics(0, 0.0, np.zeros_like(weights), 0.0)
    total = points.mapPartitions(
        lambda rows: partition_statistics(rows, weights, intercept, loss, delta)
    ).fold(zero, lambda left, right: left.merge(right))
    if total.count <= 0:
        raise ValueError("training points are empty")
    gradient = total.gradient_sum / total.count + l2 * weights
    objective = total.loss_sum / total.count + 0.5 * l2 * float(np.dot(weights, weights))
    return objective, gradient, total.intercept_gradient_sum / total.count
