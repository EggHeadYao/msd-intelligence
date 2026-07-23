from __future__ import annotations

import math
from typing import Any, Sequence

from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql import functions as F

from columns import (
    CLIPPED_CONTINUOUS_COLUMNS,
    FADE_RATIO_COLUMNS,
    KEY_CIRCULAR_COLUMNS,
    KEY_COLUMN,
    LOG_CONTINUOUS_COLUMNS,
    PASSTHROUGH_BINARY_COLUMNS,
    PASSTHROUGH_CONTINUOUS_COLUMNS,
    SCALAR_AVAILABILITY_COLUMNS,
    SEGMENT_FEATURE_COLUMNS,
    TIME_SIGNATURE_COLUMN,
    TIME_SIGNATURE_UNKNOWN_COLUMN,
    TIME_SIGNATURE_VALUES,
    build_feature_columns,
    time_signature_one_hot_column,
)

NEAR_ZERO_RANGE_EPSILON = 1e-12
SEGMENT_MEDIAN_BATCH_SIZE = 32


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
        _finite("duration") & (F.col("duration") > 0) & _finite("end_of_fade_in"),
        _finite("duration") & (F.col("duration") > 0) & _finite("start_of_fade_out"),
    )
    for name, condition in zip(SCALAR_AVAILABILITY_COLUMNS, conditions, strict=True):
        df = df.withColumn(name, condition.cast("double"))
    df = df.withColumn(
        "mode",
        F.when(F.col("has_mode") == 1.0, F.col("mode").cast("double")),
    )
    for column in ("key_confidence", "mode_confidence", "time_signature_confidence"):
        value = F.col(column).cast("double")
        df = df.withColumn(
            column,
            F.when(_finite(column), F.least(F.greatest(value, F.lit(0.0)), F.lit(1.0))),
        )
    return df


def add_fade_ratios(df: DataFrame) -> DataFrame:
    duration = F.col("duration").cast("double")
    ratios = (
        F.col("end_of_fade_in").cast("double") / duration,
        (duration - F.col("start_of_fade_out").cast("double")) / duration,
    )
    masks = ("has_fade_in_ratio", "has_fade_out_ratio")
    for column, ratio, mask in zip(FADE_RATIO_COLUMNS, ratios, masks, strict=True):
        clipped = F.least(F.greatest(ratio, F.lit(0.0)), F.lit(1.0))
        df = df.withColumn(column, F.when(F.col(mask) == 1.0, clipped))
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
        one_hot = F.when(
            F.col(TIME_SIGNATURE_COLUMN).cast("int") == F.lit(value), F.lit(1.0)
        ).otherwise(F.lit(0.0))
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
    medians = {}
    for offset in range(0, len(SEGMENT_FEATURE_COLUMNS), SEGMENT_MEDIAN_BATCH_SIZE):
        batch = SEGMENT_FEATURE_COLUMNS[offset:offset + SEGMENT_MEDIAN_BATCH_SIZE]
        medians_row = df.agg(
            *(F.percentile_approx(F.col(column).cast("double"), 0.5, 10_000).alias(column)
              for column in batch)
        ).first()
        for column in batch:
            value = medians_row[column]
            median = float(value) if value is not None else 0.0
            medians[column] = median if math.isfinite(median) else 0.0
    expressions = []
    for column in df.columns:
        if column in medians:
            valid = F.col(column).isNotNull() & ~F.isnan(F.col(column).cast("double"))
            filled = F.when(valid, F.col(column).cast("double")).otherwise(
                F.lit(medians[column])
            )
            expressions.append(filled.alias(column))
        else:
            expressions.append(F.col(column))
    return df.select(*expressions), medians


def fill_scalar_missing_values(df: DataFrame) -> tuple[DataFrame, dict[str, float]]:
    columns = (
        *KEY_CIRCULAR_COLUMNS,
        *LOG_CONTINUOUS_COLUMNS,
        *CLIPPED_CONTINUOUS_COLUMNS,
        *PASSTHROUGH_CONTINUOUS_COLUMNS,
        *FADE_RATIO_COLUMNS,
        *PASSTHROUGH_BINARY_COLUMNS,
    )
    medians_row = df.agg(
        *(F.percentile_approx(F.col(column).cast("double"), 0.5, 10_000).alias(column)
          for column in columns)
    ).first()
    medians = {}
    for column in columns:
        value = medians_row[column]
        median = float(value) if value is not None else 0.0
        medians[column] = median if math.isfinite(median) else 0.0
        current = F.col(column).cast("double")
        valid = current.isNotNull() & ~F.isnan(current) & (F.abs(current) != float("inf"))
        df = df.withColumn(column, F.when(valid, current).otherwise(F.lit(medians[column])))
    return df, medians


def drop_zero_variance_features(
    df: DataFrame,
    columns: Sequence[str],
    eps: float = NEAR_ZERO_RANGE_EPSILON,
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


def preprocess_audio_features(
    df: DataFrame,
    storage_level: StorageLevel | None = None,
) -> tuple[DataFrame, tuple[str, ...], dict[str, Any]]:
    df, segment_medians = fill_segment_missing_values(df)
    df = add_scalar_availability(df)
    df = add_fade_ratios(df)
    df = add_key_circular_features(df)
    df, time_values, time_columns = add_time_signature_one_hot(df)
    df, clip_bounds = add_log_clipped_features(df)
    df, scalar_medians = fill_scalar_missing_values(df)
    candidates = build_feature_columns(time_columns)
    if storage_level is not None:
        df = df.persist(storage_level)
    try:
        features, dropped = drop_zero_variance_features(df, candidates)
    except Exception:
        if storage_level is not None:
            df.unpersist(blocking=False)
        raise
    metadata = {
        "clip_bounds": clip_bounds,
        "dropped_features": dropped,
        "near_zero_range_epsilon": NEAR_ZERO_RANGE_EPSILON,
        "segment_medians": segment_medians,
        "scalar_medians": scalar_medians,
        "time_signature_columns": time_columns,
        "time_signature_values": time_values,
    }
    return df, features, metadata


def _require_finite_mapping(
    values: Any,
    expected_keys: Sequence[str],
    name: str,
) -> dict[str, float]:
    if not isinstance(values, dict):
        raise ValueError(f"frozen preprocessing {name} must be an object")
    if set(values) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(values))
        extra = sorted(set(values) - set(expected_keys))
        raise ValueError(
            f"frozen preprocessing {name} keys mismatch: missing={missing}, extra={extra}"
        )
    result = {key: float(values[key]) for key in expected_keys}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"frozen preprocessing {name} contains a non-finite value")
    return result


def fill_segment_missing_values_frozen(
    df: DataFrame,
    medians: Any,
) -> DataFrame:
    frozen = _require_finite_mapping(medians, SEGMENT_FEATURE_COLUMNS, "segment_medians")
    expressions = []
    for column in df.columns:
        if column in frozen:
            current = F.col(column).cast("double")
            valid = current.isNotNull() & ~F.isnan(current) & (F.abs(current) != float("inf"))
            expressions.append(
                F.when(valid, current).otherwise(F.lit(frozen[column])).alias(column)
            )
        else:
            expressions.append(F.col(column))
    return df.select(*expressions)


def add_log_clipped_features_frozen(df: DataFrame, bounds: Any) -> DataFrame:
    if not isinstance(bounds, dict) or set(bounds) != {"loudness", "tempo", "duration"}:
        raise ValueError("frozen preprocessing clip_bounds keys mismatch")
    frozen: dict[str, tuple[float, float]] = {}
    for column in ("loudness", "tempo", "duration"):
        pair = bounds[column]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"frozen preprocessing clip_bounds[{column}] must have two values")
        low, high = float(pair[0]), float(pair[1])
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"frozen preprocessing clip_bounds[{column}] is invalid")
        frozen[column] = (low, high)

    availability = {"loudness": "has_loudness", "tempo": "has_tempo", "duration": "has_duration"}
    for source, target in zip(("tempo", "duration"), LOG_CONTINUOUS_COLUMNS):
        low, high = frozen[source]
        transformed = F.log1p(F.greatest(_clip(source, low, high), F.lit(0.0)))
        df = df.withColumn(
            target,
            F.when(F.col(availability[source]) == 1.0, transformed),
        )
    low, high = frozen["loudness"]
    loudness = F.when(F.col("has_loudness") == 1.0, _clip("loudness", low, high))
    return df.withColumn(CLIPPED_CONTINUOUS_COLUMNS[0], loudness)


