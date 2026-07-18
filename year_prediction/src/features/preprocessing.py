from __future__ import annotations

import math
from typing import Any, Sequence

from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql import functions as F

from columns import (
    HAS_SEGMENTS_COLUMN,
    INPUT_COLUMNS,
    KEY_COLUMN,
    KEY_ENCODED_COLUMNS,
    MODE_COLUMN,
    SEGMENT_COLUMNS,
    TIME_SIGNATURE_COLUMN,
    TIME_SIGNATURE_UNKNOWN_COLUMN,
    TRANSFORMED_CONTINUOUS_COLUMNS,
    candidate_columns,
    time_signature_column,
)

CLIP_COLUMNS = ("tempo", "duration", "loudness")
DEFAULT_QUANTILE_ERROR = 0.001
DEFAULT_VARIANCE_THRESHOLD = 1.0e-12
UNKNOWN_CATEGORY = -1


def require_input_columns(df: DataFrame) -> None:
    missing = sorted(set(INPUT_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing feature input columns: {missing}")


def validate_binary_columns(df: DataFrame) -> None:
    invalid = df.agg(
        F.sum((F.col(MODE_COLUMN).isNull() | ~F.col(MODE_COLUMN).isin(0, 1)).cast("long")).alias(MODE_COLUMN),
        F.sum(
            (F.col(HAS_SEGMENTS_COLUMN).isNull() | ~F.col(HAS_SEGMENTS_COLUMN).isin(0, 1)).cast("long")
        ).alias(HAS_SEGMENTS_COLUMN),
    ).first()
    bad = {column: int(invalid[column]) for column in (MODE_COLUMN, HAS_SEGMENTS_COLUMN) if invalid[column]}
    if bad:
        raise ValueError(f"Invalid binary feature values: {bad}")


def finite_value(column: str) -> Column:
    value = F.col(column).cast("double")
    return F.when(
        value.isNotNull() & ~F.isnan(value) & (F.abs(value) != F.lit(float("inf"))),
        value,
    )


def fit_clip_bounds(df: DataFrame, relative_error: float) -> dict[str, list[float]]:
    expressions = []
    for column in CLIP_COLUMNS:
        valid = finite_value(column)
        if column == "tempo":
            valid = F.when(valid > 0.0, valid)
        expressions.append(valid.alias(column))
    quantile_rows = df.select(*expressions).approxQuantile(
        list(CLIP_COLUMNS), [0.01, 0.99], relative_error
    )
    bounds: dict[str, list[float]] = {}
    for column, quantiles in zip(CLIP_COLUMNS, quantile_rows):
        if len(quantiles) != 2 or not all(math.isfinite(value) for value in quantiles):
            raise ValueError(f"Cannot fit clipping bounds for {column}")
        bounds[column] = [float(quantiles[0]), float(quantiles[1])]
    return bounds


def fit_segment_means(df: DataFrame) -> dict[str, float]:
    source = df.where(F.col(HAS_SEGMENTS_COLUMN) == 1)
    row = source.agg(*(F.avg(finite_value(column)).alias(column) for column in SEGMENT_COLUMNS)).first()
    means = {}
    for column in SEGMENT_COLUMNS:
        value = row[column]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Cannot fit segment mean for {column}")
        means[column] = float(value)
    return means


def fit_time_signature_values(df: DataFrame) -> tuple[int, ...]:
    rows = (
        df.select(F.col(TIME_SIGNATURE_COLUMN).cast("int").alias(TIME_SIGNATURE_COLUMN))
        .where(F.col(TIME_SIGNATURE_COLUMN) > 0)
        .distinct()
        .orderBy(TIME_SIGNATURE_COLUMN)
        .collect()
    )
    return tuple(int(row[TIME_SIGNATURE_COLUMN]) for row in rows)


def clipped(value: Column, low: float, high: float) -> Column:
    return F.when(value.isNotNull(), F.least(F.greatest(value, F.lit(low)), F.lit(high)))


