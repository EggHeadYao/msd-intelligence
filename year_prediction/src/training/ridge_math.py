from __future__ import annotations

import math
from collections.abc import Sequence


Vector = Sequence[float]
Matrix = Sequence[Vector]


def _validate_inputs(
    features: Matrix,
    labels: Vector,
    weights: Vector,
    intercept: float,
    l2: float,
) -> None:
    if not features:
        raise ValueError("features must not be empty")
    if len(features) != len(labels):
        raise ValueError("features and labels must contain the same number of rows")
    if not weights:
        raise ValueError("weights must not be empty")
    if any(len(row) != len(weights) for row in features):
        raise ValueError("every feature row must match the weight dimension")
    values = [*labels, *weights, intercept, l2]
    values.extend(value for row in features for value in row)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Ridge inputs must be finite")
    if l2 < 0:
        raise ValueError("l2 must not be negative")


def predict(feature: Vector, weights: Vector, intercept: float) -> float:
    if len(feature) != len(weights):
        raise ValueError("feature and weight dimensions differ")
    return sum(value * weight for value, weight in zip(feature, weights)) + intercept


def ridge_loss(
    features: Matrix,
    labels: Vector,
    weights: Vector,
    intercept: float,
    l2: float,
) -> float:
    _validate_inputs(features, labels, weights, intercept, l2)
    squared_error = sum(
        (predict(row, weights, intercept) - label) ** 2
        for row, label in zip(features, labels)
    ) / len(features)
    return squared_error + l2 * sum(weight * weight for weight in weights)


def ridge_gradient(
    features: Matrix,
    labels: Vector,
    weights: Vector,
    intercept: float,
    l2: float,
) -> tuple[list[float], float]:
    _validate_inputs(features, labels, weights, intercept, l2)
    gradient = [0.0] * len(weights)
    intercept_gradient = 0.0
    for row, label in zip(features, labels):
        residual = predict(row, weights, intercept) - label
        for index, value in enumerate(row):
            gradient[index] += 2.0 * residual * value
        intercept_gradient += 2.0 * residual
    count = len(features)
    gradient = [
        value / count + 2.0 * l2 * weight
        for value, weight in zip(gradient, weights)
    ]
    return gradient, intercept_gradient / count