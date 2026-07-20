from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import reduce
from pathlib import Path
from typing import Any

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from contract import (
    ARTIST_ID,
    AUDIT_COLUMNS,
    AUDIO_FEATURE_ORDER_SHA256,
    BINARY_FEATURE_COLUMNS,
    DERIVED_SCALAR_COLUMNS,
    EXPECTED_GROUP_COUNTS,
    EXPECTED_LABELED_TRACKS,
    EXPECTED_TRACKS,
    FEATURE_CONTRACT_VERSION,
    FORBIDDEN_PREDICTOR_COLUMNS,
    GLOBAL_SCALAR_COLUMNS,
    SPLIT,
    T90_COLUMNS,
    TRACK_ID,
    YEAR,
    YEAR_EXCLUDED_COLUMNS,
    column_missing_rule,
    column_source,
    column_unit,
    full_predictor_columns,
    load_audio_contract,
    order_sha256,
    ordered_feature_groups,
    year_shared_columns,
)

FADE_TOLERANCE_SECONDS = 0.001
SPLIT_VALUES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate year-prediction feature views.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("parquets/year_prediction/features"),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("parquets/year_prediction/raw/audio_features"),
    )
    parser.add_argument(
        "--scalar",
        type=Path,
        default=Path("parquets/year_prediction/raw/songs_scalar.parquet"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("parquets/year_prediction/dataset"),
    )
    parser.add_argument("--hdf5-root", type=Path)
    parser.add_argument("--hdf5-samples", type=int, default=16)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    return parser.parse_args()


def spark_path(path: str | Path) -> str:
    text = str(path)
    return text if "://" in text else Path(text).resolve().as_uri()


def audio_paths(path: Path) -> list[str]:
    return [spark_path(item) for item in sorted(path.glob("features_*.parquet"))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def schema_types(frame: DataFrame) -> dict[str, str]:
    return {field.name: field.dataType.simpleString() for field in frame.schema.fields}


def require_schema(
    frame: DataFrame,
    columns: tuple[str, ...],
    types: dict[str, str],
    label: str,
) -> None:
    require(tuple(frame.columns) == columns, f"{label} column order differs")
    actual = schema_types(frame)
    require(actual == types, f"{label} schema differs")


def require_same(left: DataFrame, right: DataFrame, columns: tuple[str, ...], label: str) -> None:
    left_view = left.select(*columns)
    right_view = right.select(*columns)
    require(left_view.exceptAll(right_view).limit(1).count() == 0, f"{label}: unexpected rows")
    require(right_view.exceptAll(left_view).limit(1).count() == 0, f"{label}: missing rows")


def row_digest(frame: DataFrame, columns: tuple[str, ...], name: str) -> DataFrame:
    payload = F.to_json(
        F.struct(*(F.col(column) for column in columns)),
        options={"ignoreNullFields": "false"},
    )
    return frame.select(TRACK_ID, F.sha2(payload, 256).alias(name))


def require_same_values(
    left: DataFrame,
    right: DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    left_hash = row_digest(left, columns, "_left_hash")
    right_hash = row_digest(right, columns, "_right_hash")
    mismatch = (
        left_hash.join(right_hash, TRACK_ID, "inner")
        .where(F.col("_left_hash") != F.col("_right_hash"))
        .limit(1)
        .count()
    )
    require(mismatch == 0, f"{label}: values differ")


def finite(column: Column) -> Column:
    return column.isNotNull() & ~F.isnan(column) & (F.abs(column) != F.lit(float("inf")))


def clip_ratio(numerator: Column, denominator: Column) -> Column:
    return F.greatest(F.lit(0.0), F.least(F.lit(1.0), numerator / denominator))


def expected_globals(scalar: DataFrame) -> DataFrame:
    cleaned = (
        scalar.withColumn(
            "tempo",
            F.when(finite(F.col("tempo")) & (F.col("tempo") > 0), F.col("tempo")).cast(
                "double"
            ),
        )
        .withColumn("key", F.when(F.col("key").between(0, 11), F.col("key")).cast("int"))
        .withColumn("mode", F.when(F.col("mode").isin(0, 1), F.col("mode")).cast("int"))
        .withColumn(
            "time_signature",
            F.when(F.col("time_signature") > 0, F.col("time_signature")).cast("int"),
        )
    )
    duration = F.col("duration")
    fade_in = F.col("end_of_fade_in")
    fade_out = F.col("start_of_fade_out")
    valid = (
        finite(duration)
        & (duration > 0.0)
        & finite(fade_in)
        & finite(fade_out)
        & (fade_in >= 0.0)
        & (fade_out >= 0.0)
        & (fade_in <= fade_out)
        & (fade_in <= duration + FADE_TOLERANCE_SECONDS)
        & (fade_out <= duration + FADE_TOLERANCE_SECONDS)
    )
    clipped_in = F.least(fade_in, duration)
    clipped_out = F.least(fade_out, duration)
    missing = F.lit(None).cast("double")
    return (
        cleaned.withColumn(
            "fade_in_ratio",
            F.when(valid, clip_ratio(clipped_in, duration)).otherwise(missing),
        )
        .withColumn(
            "fade_out_ratio",
            F.when(valid, clip_ratio(duration - clipped_out, duration)).otherwise(missing),
        )
        .withColumn(
            "active_audio_ratio",
            F.when(valid, clip_ratio(clipped_out - clipped_in, duration)).otherwise(missing),
        )
        .select(TRACK_ID, *GLOBAL_SCALAR_COLUMNS, *DERIVED_SCALAR_COLUMNS)
    )
