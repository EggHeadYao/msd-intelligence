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

