from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from target import MAX_YEAR, MIN_YEAR, denormalize_year

MetricPartial = tuple[float, float, float, float, float, int, int]
ABSOLUTE_ERROR_COLUMN = "absolute_error_years"


@dataclass(frozen=True)
class RegressionMetrics:
    count: int
    mae_years: float
    rmse_years: float
    raw_mae_years: float
    raw_rmse_years: float
    signed_error_years: float
    raw_out_of_range_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mae_years": self.mae_years,
            "rmse_years": self.rmse_years,
            "raw_mae_years": self.raw_mae_years,
            "raw_rmse_years": self.raw_rmse_years,
            "signed_error_years": self.signed_error_years,
            "raw_out_of_range_rate": self.raw_out_of_range_rate,
        }


def normalized_to_year(value: float) -> float:
    return denormalize_year(value)


def clip_year(value: float) -> float:
    return float(min(MAX_YEAR, max(MIN_YEAR, value)))


def prediction_metric_partial(label: float, prediction: float) -> MetricPartial:
    if not math.isfinite(label) or not math.isfinite(prediction):
        raise ValueError("labels and predictions must be finite")
    target_year = normalized_to_year(label)
    raw_year = normalized_to_year(prediction)
    clipped_year = clip_year(raw_year)
    clipped_error = clipped_year - target_year
    raw_error = raw_year - target_year
    return (
        abs(clipped_error),
        clipped_error * clipped_error,
        abs(raw_error),
        raw_error * raw_error,
        clipped_error,
        int(raw_year < MIN_YEAR or raw_year > MAX_YEAR),
        1,
    )


def merge_metric_partials(left: MetricPartial, right: MetricPartial) -> MetricPartial:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def finalize_metric_partial(partial: MetricPartial) -> RegressionMetrics:
    clipped_absolute, clipped_squared, raw_absolute, raw_squared, signed, outside, count = partial
    if count <= 0:
        raise ValueError("cannot finalize empty prediction metrics")
    return RegressionMetrics(
        count=count,
        mae_years=clipped_absolute / count,
        rmse_years=math.sqrt(clipped_squared / count),
        raw_mae_years=raw_absolute / count,
        raw_rmse_years=math.sqrt(raw_squared / count),
        signed_error_years=signed / count,
        raw_out_of_range_rate=outside / count,
    )


def add_absolute_error(predictions: DataFrame) -> DataFrame:
    return predictions.withColumn(
        ABSOLUTE_ERROR_COLUMN,
        F.abs(F.col("clipped_prediction_year") - F.col("year")),
    )


def _quality_aggregations() -> list[Any]:
    clipped_error = F.col("clipped_prediction_year") - F.col("year")
    raw_error = F.col("raw_prediction_year") - F.col("year")
    return [
        F.count("*").alias("count"),
        F.countDistinct("track_id").alias("distinct_tracks"),
        F.countDistinct("artist_id").alias("distinct_artists"),
        F.avg(ABSOLUTE_ERROR_COLUMN).alias("mae_years"),
        F.sqrt(F.avg(clipped_error * clipped_error)).alias("rmse_years"),
        F.percentile_approx(ABSOLUTE_ERROR_COLUMN, 0.5, 100000).alias(
            "median_absolute_error_years"
        ),
        F.avg(F.when(F.col(ABSOLUTE_ERROR_COLUMN) <= 5.0, 1.0).otherwise(0.0)).alias(
            "within_5_years_rate"
        ),
        F.avg(F.when(F.col(ABSOLUTE_ERROR_COLUMN) <= 10.0, 1.0).otherwise(0.0)).alias(
            "within_10_years_rate"
        ),
        F.avg(clipped_error).alias("signed_error_years"),
        F.avg(F.abs(raw_error)).alias("raw_mae_years"),
        F.sqrt(F.avg(raw_error * raw_error)).alias("raw_rmse_years"),
        F.avg(
            F.when(
                (F.col("raw_prediction_year") < MIN_YEAR)
                | (F.col("raw_prediction_year") > MAX_YEAR),
                1.0,
            ).otherwise(0.0)
        ).alias("raw_out_of_range_rate"),
        F.min("raw_prediction_year").alias("minimum_raw_prediction_year"),
        F.max("raw_prediction_year").alias("maximum_raw_prediction_year"),
    ]


def _as_number(value: Any) -> int | float:
    if isinstance(value, bool) or value is None:
        raise ValueError("quality metric is missing or invalid")
    if isinstance(value, int):
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("quality metric is not finite")
    return number


def aggregate_quality_metrics(predictions: DataFrame) -> dict[str, int | float]:
    row = predictions.agg(*_quality_aggregations()).first()
    if row is None or int(row["count"]) <= 0:
        raise ValueError("cannot evaluate empty predictions")
    return {name: _as_number(value) for name, value in row.asDict().items()}


def aggregate_decade_metrics(predictions: DataFrame) -> list[dict[str, int | float]]:
    clipped_error = F.col("clipped_prediction_year") - F.col("year")
    rows = (
        predictions.withColumn(
            "decade",
            (F.floor(F.col("year") / F.lit(10)) * F.lit(10)).cast("int"),
        )
        .groupBy("decade")
        .agg(
            F.count("*").alias("count"),
            F.avg(ABSOLUTE_ERROR_COLUMN).alias("mae_years"),
            F.sqrt(F.avg(clipped_error * clipped_error)).alias("rmse_years"),
            F.avg(clipped_error).alias("signed_error_years"),
        )
        .orderBy("decade")
        .collect()
    )
    return [
        {name: _as_number(value) for name, value in row.asDict().items()}
        for row in rows
    ]
