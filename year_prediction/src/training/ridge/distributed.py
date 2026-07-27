from __future__ import annotations

from collections.abc import Sequence

from pyspark import RDD

from metrics import (
    MetricPartial,
    clip_year,
    finalize_metric_partial,
    merge_metric_partials,
    normalized_to_year,
    prediction_metric_partial,
)
from objectives import (
    Point,
    RidgeStatistics,
    finalize_ridge_partial,
    merge_ridge_partials,
    predict,
    squared_point_partial,
)


def direct_batch_statistics(
    points: RDD[Point],
    weights: Sequence[float],
    intercept: float,
    l2: float,
) -> RidgeStatistics:
    partial = points.map(
        lambda point: squared_point_partial(point, weights, intercept)
    ).reduce(merge_ridge_partials)
    return finalize_ridge_partial(partial, weights, l2)


def direct_full_batch_statistics(
    points: RDD[Point],
    weights: Sequence[float],
    intercept: float,
    l2: float,
) -> RidgeStatistics:
    return direct_batch_statistics(points, weights, intercept, l2)


def sample_mini_batch(
    points: RDD[Point],
    fraction: float,
    seed: int,
) -> RDD[Point]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("mini-batch fraction must be between zero and one")
    return points.sample(withReplacement=False, fraction=fraction, seed=seed)


def evaluate_linear_model(
    points: RDD[Point],
    weights: Sequence[float],
    intercept: float,
):
    partial: MetricPartial = points.map(
        lambda point: prediction_metric_partial(
            point[1], predict(point[0], weights, intercept)
        )
    ).reduce(merge_metric_partials)
    return finalize_metric_partial(partial)


def prediction_row(
    row: tuple[str, str, int, float, Sequence[float]],
    weights: Sequence[float],
    intercept: float,
) -> tuple[str, str, int, float, float, float, float]:
    track_id, artist_id, year, label, features = row
    normalized_prediction = predict(features, weights, intercept)
    raw_year = normalized_to_year(normalized_prediction)
    return (
        track_id,
        artist_id,
        year,
        label,
        normalized_prediction,
        raw_year,
        clip_year(raw_year),
    )
