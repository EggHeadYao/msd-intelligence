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


def transform_before_imputation(df: DataFrame, state: dict[str, Any]) -> DataFrame:
    bounds = state["clip_bounds"]
    replacements = {column: finite_value(column) for column in ("danceability", "energy")}
    for column in SEGMENT_COLUMNS:
        valid = (F.col(HAS_SEGMENTS_COLUMN) == 1) & finite_value(column).isNotNull()
        replacements[column] = F.when(valid, finite_value(column)).otherwise(
            F.lit(state["segment_means"][column])
        )
    loudness = finite_value("loudness")
    tempo = F.when(finite_value("tempo") > 0.0, finite_value("tempo"))
    duration = finite_value("duration")
    clipped_duration = clipped(duration, *bounds["duration"])
    return df.select(
        *(replacements.get(column, F.col(column)).alias(column) for column in df.columns),
        clipped(loudness, *bounds["loudness"]).alias("loudness_clipped"),
        F.log1p(clipped(tempo, *bounds["tempo"])).alias("tempo_log"),
        F.when(
            clipped_duration.isNotNull(),
            F.log1p(F.greatest(clipped_duration, F.lit(0.0))),
        ).alias("duration_log"),
    )


def fit_imputation_means(df: DataFrame) -> dict[str, float]:
    row = df.agg(*(F.avg(F.col(column)).alias(column) for column in TRANSFORMED_CONTINUOUS_COLUMNS)).first()
    means = {}
    for column in TRANSFORMED_CONTINUOUS_COLUMNS:
        value = row[column]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Cannot fit imputation mean for {column}")
        means[column] = float(value)
    return means


def add_encodings(df: DataFrame, state: dict[str, Any]) -> DataFrame:
    known_key = F.col(KEY_COLUMN).cast("int").between(0, 11)
    angle = F.col(KEY_COLUMN).cast("double") * F.lit(2.0 * math.pi / 12.0)
    values = tuple(state["time_signature_values"])
    known_time = F.col(TIME_SIGNATURE_COLUMN).cast("int").isin(*values) if values else F.lit(False)
    replacements = {
        KEY_COLUMN: F.when(known_key, F.col(KEY_COLUMN).cast("int")).otherwise(UNKNOWN_CATEGORY),
        TIME_SIGNATURE_COLUMN: F.when(
            known_time, F.col(TIME_SIGNATURE_COLUMN).cast("int")
        ).otherwise(UNKNOWN_CATEGORY),
    }
    return df.select(
        *(replacements.get(column, F.col(column)).alias(column) for column in df.columns),
        F.when(known_key, F.sin(angle)).otherwise(0.0).alias(KEY_ENCODED_COLUMNS[0]),
        F.when(known_key, F.cos(angle)).otherwise(0.0).alias(KEY_ENCODED_COLUMNS[1]),
        F.when(known_key, 0.0).otherwise(1.0).alias(KEY_ENCODED_COLUMNS[2]),
        *(
            (F.col(TIME_SIGNATURE_COLUMN).cast("int") == value)
            .cast("double")
            .alias(time_signature_column(value))
            for value in values
        ),
        F.when(known_time, 0.0).otherwise(1.0).alias(TIME_SIGNATURE_UNKNOWN_COLUMN),
    )


def transform_features(df: DataFrame, state: dict[str, Any]) -> DataFrame:
    require_input_columns(df)
    result = transform_before_imputation(df, state)
    replacements = {
        column: F.coalesce(F.col(column).cast("double"), F.lit(mean))
        for column, mean in state["imputation_means"].items()
    }
    replacements[MODE_COLUMN] = F.col(MODE_COLUMN).cast("double")
    replacements[HAS_SEGMENTS_COLUMN] = F.col(HAS_SEGMENTS_COLUMN).cast("double")
    imputed = result.select(
        *(replacements.get(column, F.col(column)).alias(column) for column in result.columns)
    )
    return add_encodings(imputed, state)


def feature_statistics(df: DataFrame, columns: Sequence[str]) -> dict[str, dict[str, float]]:
    row = df.agg(
        *(
            expression
            for column in columns
            for expression in (
                F.avg(F.col(column)).alias(f"{column}__mean"),
                F.stddev_samp(F.col(column)).alias(f"{column}__std"),
            )
        )
    ).first()
    result = {}
    for column in columns:
        mean = float(row[f"{column}__mean"])
        std_value = row[f"{column}__std"]
        result[column] = {"mean": mean, "std": float(std_value) if std_value is not None else 0.0}
    return result


def fit_feature_contract(
    train: DataFrame,
    quantile_error: float = DEFAULT_QUANTILE_ERROR,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> dict[str, Any]:
    require_input_columns(train)
    validate_binary_columns(train)
    state: dict[str, Any] = {
        "clip_quantiles": [0.01, 0.99],
        "clip_relative_error": quantile_error,
        "clip_bounds": fit_clip_bounds(train, quantile_error),
        "segment_means": fit_segment_means(train),
        "time_signature_values": list(fit_time_signature_values(train)),
        "unknown_category": UNKNOWN_CATEGORY,
    }
    prepared = transform_before_imputation(train, state)
    state["imputation_means"] = fit_imputation_means(prepared)
    transformed = transform_features(train, state)
    candidates = candidate_columns(tuple(state["time_signature_values"]))
    statistics = feature_statistics(transformed, candidates)
    retained = tuple(
        column for column in candidates if statistics[column]["std"] ** 2 > variance_threshold
    )
    if not retained:
        raise ValueError("All candidate features were removed by variance filtering")
    state.update(
        {
            "variance_threshold": variance_threshold,
            "candidate_columns": list(candidates),
            "retained_columns": list(retained),
            "removed_columns": [column for column in candidates if column not in retained],
            "candidate_statistics": statistics,
            "scaler_mean": [statistics[column]["mean"] for column in retained],
            "scaler_std": [statistics[column]["std"] for column in retained],
        }
    )
    return state
