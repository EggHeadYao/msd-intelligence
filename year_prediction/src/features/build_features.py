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


