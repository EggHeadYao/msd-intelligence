from __future__ import annotations

import math
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

MIN_YEAR = 1922.0
MAX_YEAR = 2011.0


def add_prediction_columns(
    frame: DataFrame, source: str = "prediction"
) -> DataFrame:
    raw = F.col(source).cast("double")
    clipped = F.least(F.lit(MAX_YEAR), F.greatest(F.lit(MIN_YEAR), raw))
    return (
        frame.withColumn("raw_prediction_year", raw)
        .withColumn("clipped_prediction_year", clipped)
        .withColumn(
            "absolute_error_years",
            F.abs(F.col("clipped_prediction_year") - F.col("year")),
        )
    )


def _number(value: Any) -> int | float:
    if value is None or isinstance(value, bool):
        raise ValueError("metric is missing")
    if isinstance(value, int):
        return value
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("metric is not finite")
    return result


def regression_metrics(predictions: DataFrame) -> dict[str, int | float]:
    clipped_error = F.col("clipped_prediction_year") - F.col("year")
    raw_error = F.col("raw_prediction_year") - F.col("year")
    row = predictions.agg(
        F.count("*").alias("count"),
        F.countDistinct("track_id").alias("distinct_tracks"),
        F.countDistinct("artist_id").alias("distinct_artists"),
        F.avg("absolute_error_years").alias("mae_years"),
        F.sqrt(F.avg(clipped_error * clipped_error)).alias("rmse_years"),
        F.percentile_approx("absolute_error_years", 0.5, 100000).alias(
            "median_absolute_error_years"
        ),
        F.avg(F.when(F.col("absolute_error_years") <= 5.0, 1.0).otherwise(0.0)).alias(
            "within_5_years_rate"
        ),
        F.avg(F.when(F.col("absolute_error_years") <= 10.0, 1.0).otherwise(0.0)).alias(
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
    ).first()
    if row is None or int(row["count"]) <= 0:
        raise ValueError("cannot evaluate empty predictions")
    return {name: _number(value) for name, value in row.asDict().items()}
