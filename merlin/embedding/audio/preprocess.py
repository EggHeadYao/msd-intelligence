from __future__ import annotations

import math
from typing import Any, Sequence

from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql import functions as F

from columns import (
    CLIPPED_CONTINUOUS_COLUMNS,
    HAS_SEGMENTS_COLUMN,
    KEY_CIRCULAR_COLUMNS,
    KEY_COLUMN,
    LOG_CONTINUOUS_COLUMNS,
    SEGMENT_FEATURE_COLUMNS,
    TIME_SIGNATURE_COLUMN,
    build_feature_columns,
    time_signature_one_hot_column,
)


def add_key_circular_features(df: DataFrame) -> DataFrame:
    angle = F.col(KEY_COLUMN).cast("double") * F.lit(2.0 * math.pi / 12.0)
    return df.withColumn(KEY_CIRCULAR_COLUMNS[0], F.sin(angle)).withColumn(
        KEY_CIRCULAR_COLUMNS[1], F.cos(angle)
    )


def add_time_signature_one_hot(
    df: DataFrame,
    values: Sequence[int] | None = None,
) -> tuple[DataFrame, tuple[int, ...], tuple[str, ...]]:
    if values is None:
        rows = (
            df.select(TIME_SIGNATURE_COLUMN)
            .where(F.col(TIME_SIGNATURE_COLUMN).isNotNull())
            .distinct()
            .collect()
        )
        values = sorted(int(row[0]) for row in rows)
    values = tuple(int(value) for value in values)
    columns = tuple(time_signature_one_hot_column(value) for value in values)
    for value, column in zip(values, columns):
        one_hot = (F.col(TIME_SIGNATURE_COLUMN).cast("int") == F.lit(value)).cast("double")
        df = df.withColumn(column, one_hot)
    return df, values, columns


def _bounds(df: DataFrame, column: str) -> tuple[float, float]:
    quantiles = df.approxQuantile(column, [0.01, 0.99], 0.001)
    if len(quantiles) < 2:
        return 0.0, 0.0
    low, high = float(quantiles[0]), float(quantiles[1])
    if not math.isfinite(low) or not math.isfinite(high):
        return 0.0, 0.0
    return low, high


def _clip(column: str, low: float, high: float) -> Column:
    return F.least(F.greatest(F.col(column).cast("double"), F.lit(low)), F.lit(high))


def add_log_clipped_features(df: DataFrame) -> tuple[DataFrame, dict[str, tuple[float, float]]]:
    thresholds = {column: _bounds(df, column) for column in ("tempo", "duration", "loudness")}
    for source, target in zip(("tempo", "duration"), LOG_CONTINUOUS_COLUMNS):
        low, high = thresholds[source]
        df = df.withColumn(target, F.log1p(F.greatest(_clip(source, low, high), F.lit(0.0))))
    low, high = thresholds["loudness"]
    return df.withColumn(CLIPPED_CONTINUOUS_COLUMNS[0], _clip("loudness", low, high)), thresholds


