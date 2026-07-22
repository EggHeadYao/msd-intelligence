from __future__ import annotations

import math

import numpy as np

from .objective import distributed_statistics


def validation_metrics(points, weights: np.ndarray, intercept: float) -> dict:
    def error(row):
        raw = 1922.0 + 89.0 * float(np.dot(row[4], weights) + intercept)
        prediction = min(2011.0, max(1922.0, raw))
        difference = prediction - row[2]
        return 1, abs(difference), difference * difference

    count, absolute, squared = points.map(error).fold(
        (0, 0.0, 0.0),
        lambda left, right: tuple(a + b for a, b in zip(left, right)),
    )
    if count <= 0:
        raise ValueError("validation points are empty")
    return {"count": count, "mae_years": absolute / count, "rmse_years": math.sqrt(squared / count)}


def fit(points, validation, dimension: int, train_mean: float, config: dict) -> dict:
    weights = np.zeros(dimension, dtype=np.float64)
    intercept = float(train_mean)
    history = []
    best = None
    stale = 0
    delta = float(config.get("huber_delta_years", 1.0)) / 89.0
    for iteration in range(1, int(config["max_iterations"]) + 1):
        objective, gradient, intercept_gradient = distributed_statistics(
            points, weights, intercept, config["loss"], delta, float(config["l2"])
        )
        norm = float(math.sqrt(np.dot(gradient, gradient) + intercept_gradient**2))
        weights = weights - float(config["learning_rate"]) * gradient
        intercept -= float(config["learning_rate"]) * intercept_gradient
        metrics = None
        if iteration % int(config["validation_interval"]) == 0:
            metrics = validation_metrics(validation, weights, intercept)
            if best is None or metrics["mae_years"] < best[0] - float(config["early_stopping_min_delta"]):
                best = metrics["mae_years"], iteration, weights.copy(), intercept
                stale = 0
            else:
                stale += 1
        history.append(
            {"iteration": iteration, "objective": objective, "gradient_norm": norm, "validation": metrics}
        )
        if norm <= float(config["gradient_tolerance"]):
            break
        if int(config["early_stopping_patience"]) and stale >= int(config["early_stopping_patience"]):
            break
    if best is not None and bool(config.get("restore_best_weights", True)):
        _, best_iteration, weights, intercept = best
    else:
        best_iteration = None
    return {
        "weights": weights, "intercept": intercept, "history": history,
        "best_iteration": best_iteration, "metrics": validation_metrics(validation, weights, intercept),
    }
