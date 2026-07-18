from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from columns import (
    ARTIST_ID,
    HAS_SEGMENTS_COLUMN,
    KEY_COLUMN,
    MAX_YEAR,
    MIN_YEAR,
    MODE_COLUMN,
    NORMALIZED_YEAR,
    SPLIT,
    TIME_SIGNATURE_COLUMN,
    TRACK_ID,
    TRAIN,
    YEAR,
)
from preprocessing import fit_feature_contract, transform_features, validate_binary_columns
from views import build_engineered_view, build_linear_view, engineered_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate year-prediction feature views.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("parquets/year_prediction/dataset"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("parquets/year_prediction/features/v1"),
    )
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    return parser.parse_args()


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def schema_types(df: DataFrame) -> dict[str, str]:
    return {field.name: field.dataType.simpleString() for field in df.schema.fields}

