from __future__ import annotations

import math
from collections.abc import Sequence


def gradient_norm(gradient: Sequence[float], intercept_gradient: float) -> float:
    return math.sqrt(
        sum(value * value for value in gradient)
        + intercept_gradient * intercept_gradient
    )


def gradient_step(
    weights: Sequence[float],
    intercept: float,
    gradient: Sequence[float],
    intercept_gradient: float,
    learning_rate: float,
) -> tuple[list[float], float]:
    if len(weights) != len(gradient):
        raise ValueError("weight and gradient dimensions differ")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    values = [*weights, intercept, *gradient, intercept_gradient]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("optimizer inputs must be finite")
    updated_weights = [
        weight - learning_rate * value
        for weight, value in zip(weights, gradient)
    ]
    updated_intercept = intercept - learning_rate * intercept_gradient
    if not all(math.isfinite(value) for value in [*updated_weights, updated_intercept]):
        raise ValueError("optimizer update produced a non-finite parameter")
    return updated_weights, updated_intercept
