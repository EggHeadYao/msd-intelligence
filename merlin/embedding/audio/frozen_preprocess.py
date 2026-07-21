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

