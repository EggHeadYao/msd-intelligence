from __future__ import annotations

import math
from typing import Any, Sequence

from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql import functions as F

from columns import (
    CLIPPED_CONTINUOUS_COLUMNS,
    KEY_CIRCULAR_COLUMNS,
    KEY_COLUMN,
    LOG_CONTINUOUS_COLUMNS,
    SCALAR_AVAILABILITY_COLUMNS,
    SEGMENT_FEATURE_COLUMNS,
    TIME_SIGNATURE_COLUMN,
    TIME_SIGNATURE_UNKNOWN_COLUMN,
    TIME_SIGNATURE_VALUES,
    build_feature_columns,
    time_signature_one_hot_column,
)


def _finite(column: str) -> Column:
    value = F.col(column).cast("double")
    return value.isNotNull() & ~F.isnan(value) & (F.abs(value) != float("inf"))


def add_scalar_availability(df: DataFrame) -> DataFrame:
    key_valid = _finite(KEY_COLUMN) & F.col(KEY_COLUMN).cast("int").between(0, 11)
    key_valid = key_valid & _finite("key_confidence") & (F.col("key_confidence") > 0)
    mode_valid = F.col("mode").cast("int").isin(0, 1)
    mode_valid = mode_valid & _finite("mode_confidence") & (F.col("mode_confidence") > 0)
    meter_valid = F.col(TIME_SIGNATURE_COLUMN).cast("int").isin(*TIME_SIGNATURE_VALUES)
    meter_valid = meter_valid & _finite("time_signature_confidence")
    meter_valid = meter_valid & (F.col("time_signature_confidence") > 0)
    conditions = (
        _finite("loudness"),
        _finite("tempo") & (F.col("tempo") > 0),
        _finite("duration") & (F.col("duration") > 0),
        key_valid,
        mode_valid,
        meter_valid,
    )
    for name, condition in zip(SCALAR_AVAILABILITY_COLUMNS, conditions, strict=True):
        df = df.withColumn(name, condition.cast("double"))
    return df


def add_key_circular_features(df: DataFrame) -> DataFrame:
    key = F.col(KEY_COLUMN).cast("double")
    confidence = F.col("key_confidence").cast("double")
    valid = key.between(0.0, 11.0) & (confidence > 0.0) & ~F.isnan(confidence)
    angle = key * F.lit(2.0 * math.pi / 12.0)
    return df.withColumn(
        KEY_CIRCULAR_COLUMNS[0], F.when(valid, F.sin(angle)).otherwise(F.lit(0.0))
    ).withColumn(
        KEY_CIRCULAR_COLUMNS[1], F.when(valid, F.cos(angle)).otherwise(F.lit(0.0))
    )


def add_time_signature_one_hot(
    df: DataFrame,
    values: Sequence[int] | None = None,
) -> tuple[DataFrame, tuple[int, ...], tuple[str, ...]]:
    if values is None:
        values = TIME_SIGNATURE_VALUES
    values = tuple(int(value) for value in values)
    columns = tuple(time_signature_one_hot_column(value) for value in values)
    for value, column in zip(values, columns):
        one_hot = (F.col(TIME_SIGNATURE_COLUMN).cast("int") == F.lit(value)).cast("double")
        df = df.withColumn(column, one_hot)
    meter = F.col(TIME_SIGNATURE_COLUMN).cast("int")
    known = meter.isin(*values)
    df = df.withColumn(
        TIME_SIGNATURE_UNKNOWN_COLUMN,
        F.when(known, F.lit(0.0)).otherwise(F.lit(1.0)),
    )
    return df, values, (*columns, TIME_SIGNATURE_UNKNOWN_COLUMN)


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
    availability = {"loudness": "has_loudness", "tempo": "has_tempo", "duration": "has_duration"}
    thresholds = {
        column: _bounds(df.where(F.col(availability[column]) == 1.0), column)
        for column in availability
    }
    for source, target in zip(("tempo", "duration"), LOG_CONTINUOUS_COLUMNS):
        low, high = thresholds[source]
        transformed = F.log1p(F.greatest(_clip(source, low, high), F.lit(0.0)))
        df = df.withColumn(
            target,
            F.when(F.col(availability[source]) == 1.0, transformed),
        )
    low, high = thresholds["loudness"]
    loudness = F.when(F.col("has_loudness") == 1.0, _clip("loudness", low, high))
    return df.withColumn(CLIPPED_CONTINUOUS_COLUMNS[0], loudness), thresholds


def fill_segment_missing_values(df: DataFrame) -> tuple[DataFrame, dict[str, float]]:
    means_row = df.agg(
        *(F.avg(F.col(column).cast("double")).alias(column) for column in SEGMENT_FEATURE_COLUMNS)
    ).first()
    means = {}
    for column in SEGMENT_FEATURE_COLUMNS:
        value = means_row[column]
        mean = float(value) if value is not None else 0.0
        means[column] = mean if math.isfinite(mean) else 0.0
    expressions = []
    for column in df.columns:
        if column in means:
            valid = F.col(column).isNotNull() & ~F.isnan(F.col(column).cast("double"))
            filled = F.when(valid, F.col(column).cast("double")).otherwise(
                F.lit(means[column])
            )
            expressions.append(filled.alias(column))
        else:
            expressions.append(F.col(column))
    return df.select(*expressions), means


def drop_zero_variance_features(
    df: DataFrame,
    columns: Sequence[str],
    eps: float = 0.0,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    aggregates = [
        item
        for column in columns
        for item in (F.min(column).alias(f"{column}__min"), F.max(column).alias(f"{column}__max"))
    ]
    stats = df.agg(*aggregates).first()
    kept = []
    for column in columns:
        low, high = stats[f"{column}__min"], stats[f"{column}__max"]
        if low is not None and high is not None and abs(float(high) - float(low)) > eps:
            kept.append(column)
    kept = tuple(kept)
    dropped = tuple(column for column in columns if column not in kept)
    return kept, dropped


def preprocess_audio_features(df: DataFrame) -> tuple[DataFrame, tuple[str, ...], dict[str, Any]]:
    df, segment_means = fill_segment_missing_values(df)
    df = add_scalar_availability(df)
    df = add_key_circular_features(df)
    df, time_values, time_columns = add_time_signature_one_hot(df)
    df, clip_bounds = add_log_clipped_features(df)
    candidates = build_feature_columns(time_columns)
    features, dropped = drop_zero_variance_features(df, candidates)
    metadata = {
        "clip_bounds": clip_bounds,
        "dropped_features": dropped,
        "segment_means": segment_means,
        "time_signature_columns": time_columns,
        "time_signature_values": time_values,
    }
    return df, features, metadata
