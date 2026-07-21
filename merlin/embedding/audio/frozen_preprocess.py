from __future__ import annotations

import math
from typing import Any, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from columns import (
    CLIPPED_CONTINUOUS_COLUMNS,
    FADE_RATIO_COLUMNS,
    KEY_CIRCULAR_COLUMNS,
    LOG_CONTINUOUS_COLUMNS,
    PASSTHROUGH_BINARY_COLUMNS,
    PASSTHROUGH_CONTINUOUS_COLUMNS,
    SEGMENT_FEATURE_COLUMNS,
    build_feature_columns,
)
from preprocess import (
    _clip,
    add_fade_ratios,
    add_key_circular_features,
    add_scalar_availability,
    add_time_signature_one_hot,
)


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


def fill_scalar_missing_values_frozen(df: DataFrame, medians: Any) -> DataFrame:
    columns = (
        *KEY_CIRCULAR_COLUMNS,
        *LOG_CONTINUOUS_COLUMNS,
        *CLIPPED_CONTINUOUS_COLUMNS,
        *PASSTHROUGH_CONTINUOUS_COLUMNS,
        *FADE_RATIO_COLUMNS,
        *PASSTHROUGH_BINARY_COLUMNS,
    )
    frozen = _require_finite_mapping(medians, columns, "scalar_medians")
    for column in columns:
        current = F.col(column).cast("double")
        valid = current.isNotNull() & ~F.isnan(current) & (F.abs(current) != float("inf"))
        df = df.withColumn(
            column,
            F.when(valid, current).otherwise(F.lit(frozen[column])),
        )
    return df

