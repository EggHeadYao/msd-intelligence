from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


Vector = Sequence[float]
Point = tuple[Vector, float]
RidgePartial = tuple[float, list[float], float, int]


@dataclass(frozen=True)
class RidgeStatistics:
    objective: float
    gradient: list[float]
    intercept_gradient: float
    count: int


def predict(features: Vector, weights: Vector, intercept: float) -> float:
    if len(features) != len(weights):
        raise ValueError("feature and weight dimensions differ")
    return sum(value * weight for value, weight in zip(features, weights)) + intercept


def squared_point_partial(point: Point, weights: Vector, intercept: float) -> RidgePartial:
    features, label = point
    residual = predict(features, weights, intercept) - label
    return (
        residual * residual,
        [2.0 * residual * value for value in features],
        2.0 * residual,
        1,
    )


def merge_ridge_partials(left: RidgePartial, right: RidgePartial) -> RidgePartial:
    if len(left[1]) != len(right[1]):
        raise ValueError("partial gradient dimensions differ")
    return (
        left[0] + right[0],
        [a + b for a, b in zip(left[1], right[1])],
        left[2] + right[2],
        left[3] + right[3],
    )


def finalize_ridge_partial(
    partial: RidgePartial,
    weights: Vector,
    l2: float,
) -> RidgeStatistics:
    squared_error, gradient_sum, intercept_gradient_sum, count = partial
    if count <= 0:
        raise ValueError("cannot finalize an empty Ridge partial")
    if len(gradient_sum) != len(weights):
        raise ValueError("gradient and weight dimensions differ")
    if l2 < 0.0 or not math.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")
    objective = squared_error / count + l2 * sum(weight * weight for weight in weights)
    gradient = [
        value / count + 2.0 * l2 * weight
        for value, weight in zip(gradient_sum, weights)
    ]
    return RidgeStatistics(
        objective=objective,
        gradient=gradient,
        intercept_gradient=intercept_gradient_sum / count,
        count=count,
    )
