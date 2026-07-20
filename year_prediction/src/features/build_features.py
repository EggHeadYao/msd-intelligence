from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from contract import (
    ARTIST_ID,
    AUDIT_COLUMNS,
    AUDIO_FEATURE_COUNT,
    DERIVED_SCALAR_COLUMNS,
    EXPECTED_LABELED_TRACKS,
    EXPECTED_TRACKS,
    FEATURE_CONTRACT_VERSION,
    FORBIDDEN_PREDICTOR_COLUMNS,
    GLOBAL_SCALAR_COLUMNS,
    SPLIT,
    T90_COLUMNS,
    TRACK_ID,
    YEAR,
    column_missing_rule,
    column_source,
    column_unit,
    full_predictor_columns,
    load_audio_contract,
    order_sha256,
    ordered_feature_groups,
    year_shared_columns,
)
from full_tabular import FADE_TOLERANCE_SECONDS, build_full_tabular
from t90 import build_t90

LABEL_COLUMNS = (TRACK_ID, ARTIST_ID, YEAR, SPLIT)
LABEL_TYPES = {TRACK_ID: "string", ARTIST_ID: "string", YEAR: "int", SPLIT: "string"}
SCALAR_TYPES = {
    TRACK_ID: "string",
    ARTIST_ID: "string",
    YEAR: "int",
    **{column: "double" for column in GLOBAL_SCALAR_COLUMNS},
    "key": "int",
    "mode": "int",
    "time_signature": "int",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build year-prediction feature views.")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parquets/year_prediction/features"),
    )
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


def require_types(frame: DataFrame, expected: dict[str, str], label: str) -> None:
    actual = schema_types(frame)
    missing = set(expected) - set(actual)
    require(not missing, f"{label} is missing columns: {sorted(missing)}")
    wrong = {column: actual[column] for column in expected if actual[column] != expected[column]}
    require(not wrong, f"{label} has unexpected types: {wrong}")


def schema_payload(frame: DataFrame) -> list[dict[str, object]]:
    return [
        {
            "name": field.name,
            "type": field.dataType.simpleString(),
            "nullable": field.nullable,
        }
        for field in frame.schema.fields
    ]


def build_metadata(scalar: DataFrame, labels: DataFrame) -> DataFrame:
    labeled = labels.select(
        TRACK_ID,
        F.col(ARTIST_ID).alias("_label_artist_id"),
        F.col(YEAR).alias("_label_year"),
        F.col(SPLIT).alias("_label_split"),
    )
    joined = scalar.select(
        TRACK_ID,
        ARTIST_ID,
        YEAR,
        *GLOBAL_SCALAR_COLUMNS,
    ).join(labeled, TRACK_ID, "left")
    label_present = F.col("_label_split").isNotNull()
    invalid = (
        (F.col(YEAR).isNotNull() != label_present)
        | (
            label_present
            & (
                ~F.col(ARTIST_ID).eqNullSafe(F.col("_label_artist_id"))
                | ~F.col(YEAR).eqNullSafe(F.col("_label_year"))
            )
        )
    )
    summary = joined.agg(
        F.count("*").alias("rows"),
        F.countDistinct(TRACK_ID).alias("tracks"),
        F.sum(F.when(label_present, 1).otherwise(0)).alias("labeled"),
        F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid"),
    ).first()
    require(int(summary["rows"]) == EXPECTED_TRACKS, "metadata row count differs")
    require(int(summary["tracks"]) == EXPECTED_TRACKS, "metadata track IDs are duplicated")
    require(int(summary["labeled"]) == EXPECTED_LABELED_TRACKS, "metadata label count differs")
    require(int(summary["invalid"]) == 0, "dataset labels disagree with scalar data")
    return joined.select(
        TRACK_ID,
        ARTIST_ID,
        YEAR,
        F.col("_label_split").alias(SPLIT),
        *GLOBAL_SCALAR_COLUMNS,
    )


def predictor_metadata(columns: tuple[str, ...], types: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "name": column,
            "type": types[column],
            "source": column_source(column),
            "unit": column_unit(column),
            "missing": column_missing_rule(column),
        }
        for column in columns
    ]


