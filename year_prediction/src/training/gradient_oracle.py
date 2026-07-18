from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pyspark import SparkContext
from pyspark.sql import SparkSession

from ridge_math import (
    finite_difference_gradient,
    gradient_step,
    relative_error,
    ridge_gradient,
    ridge_loss,
)


Partial = tuple[float, list[float], float, int]


def default_fixture_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "ridge_oracle.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Ridge gradient contract.")
    parser.add_argument("--fixture", type=Path, default=default_fixture_path())
    return parser.parse_args()


def load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def ship_oracle_modules(context: SparkContext) -> None:
    module_dir = Path(__file__).resolve().parent
    context.addPyFile(str(module_dir / "ridge_math.py"))
    context.addPyFile(str(module_dir / "gradient_oracle.py"))


def _partition_partial(
    rows: Iterable[tuple[Sequence[float], float]],
    weights: Sequence[float],
    intercept: float,
) -> Iterable[Partial]:
    gradient = [0.0] * len(weights)
    squared_error = 0.0
    intercept_gradient = 0.0
    count = 0
    for feature, label in rows:
        residual = sum(x * weight for x, weight in zip(feature, weights)) + intercept - label
        squared_error += residual * residual
        for index, value in enumerate(feature):
            gradient[index] += 2.0 * residual * value
        intercept_gradient += 2.0 * residual
        count += 1
    if count:
        yield squared_error, gradient, intercept_gradient, count


def _merge_partials(left: Partial, right: Partial) -> Partial:
    return (
        left[0] + right[0],
        [a + b for a, b in zip(left[1], right[1])],
        left[2] + right[2],
        left[3] + right[3],
    )


def spark_ridge_statistics(
    context: SparkContext,
    features: Sequence[Sequence[float]],
    labels: Sequence[float],
    weights: Sequence[float],
    intercept: float,
    l2: float,
    partitions: int,
) -> tuple[float, list[float], float, int]:
    if partitions <= 0:
        raise ValueError("partitions must be positive")
    rows = list(zip(features, labels))
    partial = context.parallelize(rows, partitions).mapPartitions(
        lambda iterator: _partition_partial(iterator, weights, intercept)
    ).reduce(_merge_partials)
    squared_error, gradient_sum, intercept_sum, count = partial
    loss = squared_error / count + l2 * sum(weight * weight for weight in weights)
    gradient = [
        value / count + 2.0 * l2 * weight
        for value, weight in zip(gradient_sum, weights)
    ]
    return loss, gradient, intercept_sum / count, count


def require_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{label} differs: expected={expected}, actual={actual}")



def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("YearPredictionRidgeGradientOracle").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        result = run_oracle(spark.sparkContext, load_fixture(args.fixture))
        print(json.dumps(result, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
