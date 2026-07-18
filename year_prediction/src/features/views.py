from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from columns import (
    ARTIST_ID,
    AUDIT_CATEGORY_COLUMNS,
    MAX_YEAR,
    MIN_YEAR,
    NORMALIZED_YEAR,
    SPLIT,
    TRACK_ID,
    YEAR,
)


def engineered_columns(state: dict[str, Any]) -> tuple[str, ...]:
    return (
        TRACK_ID,
        ARTIST_ID,
        YEAR,
        NORMALIZED_YEAR,
        *AUDIT_CATEGORY_COLUMNS,
        *state["candidate_columns"],
        SPLIT,
    )


def build_engineered_view(transformed: DataFrame, state: dict[str, Any]) -> DataFrame:
    normalized = (F.col(YEAR).cast("double") - F.lit(float(MIN_YEAR))) / F.lit(float(MAX_YEAR - MIN_YEAR))
    return transformed.withColumn(NORMALIZED_YEAR, normalized).select(*engineered_columns(state))


def scaled_feature_array(engineered: DataFrame, state: dict[str, Any]):
    means = state["scaler_mean"]
    standard_deviations = state["scaler_std"]
    expressions = [
        (F.col(column).cast("double") - F.lit(float(mean))) / F.lit(float(std))
        for column, mean, std in zip(state["retained_columns"], means, standard_deviations)
    ]
    return F.array(*expressions)


def build_linear_view(engineered: DataFrame, state: dict[str, Any]) -> DataFrame:
    return engineered.select(
        TRACK_ID,
        ARTIST_ID,
        YEAR,
        NORMALIZED_YEAR,
        scaled_feature_array(engineered, state).alias("features"),
        SPLIT,
    )

