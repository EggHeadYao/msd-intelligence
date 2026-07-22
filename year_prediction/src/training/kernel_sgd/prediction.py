from __future__ import annotations

import sys
from typing import Iterable, Iterator

import numpy as np
from pyspark import cloudpickle
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from .features import Point

cloudpickle.register_pickle_by_value(sys.modules[__name__])

MIN_YEAR = 1922.0
MAX_YEAR = 2011.0
YEAR_SPAN = MAX_YEAR - MIN_YEAR
SCHEMA = StructType(
    [
        StructField("track_id", StringType(), False),
        StructField("artist_id", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("normalized_year", DoubleType(), False),
        StructField("normalized_prediction", DoubleType(), False),
        StructField("raw_prediction_year", DoubleType(), False),
        StructField("clipped_prediction_year", DoubleType(), False),
        StructField("absolute_error_years", DoubleType(), False),
    ]
)


def prediction_partition(
    rows: Iterable[Point], weights: np.ndarray, intercept: float
) -> Iterator[tuple]:
    for track, artist, year, label, features in rows:
        normalized = float(np.dot(features, weights) + intercept)
        raw_year = MIN_YEAR + YEAR_SPAN * normalized
        clipped = min(MAX_YEAR, max(MIN_YEAR, raw_year))
        yield track, artist, year, label, normalized, raw_year, clipped, abs(clipped - year)


def quality_metrics(predictions: DataFrame) -> dict:
    row = predictions.agg(
        F.count("*").alias("count"), F.countDistinct("track_id").alias("distinct_tracks"),
        F.countDistinct("artist_id").alias("distinct_artists"),
        F.avg("absolute_error_years").alias("mae_years"),
        F.sqrt(F.avg(F.pow(F.col("clipped_prediction_year") - F.col("year"), 2.0))).alias("rmse_years"),
        F.percentile_approx("absolute_error_years", 0.5, 10000).alias("median_absolute_error_years"),
        F.avg(F.when(F.col("absolute_error_years") <= 5.0, 1.0).otherwise(0.0)).alias(
            "within_5_years_rate"
        ),
        F.avg(F.when(F.col("absolute_error_years") <= 10.0, 1.0).otherwise(0.0)).alias(
            "within_10_years_rate"
        ),
        F.avg(
            F.when((F.col("raw_prediction_year") < MIN_YEAR) | (F.col("raw_prediction_year") > MAX_YEAR), 1.0)
            .otherwise(0.0)
        ).alias("raw_out_of_range_rate"),
    ).first()
    return row.asDict()
