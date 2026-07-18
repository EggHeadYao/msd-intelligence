from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from columns import INPUT_COLUMNS, SPLIT, TRAIN
from preprocessing import (
    DEFAULT_QUANTILE_ERROR,
    DEFAULT_VARIANCE_THRESHOLD,
    fit_feature_contract,
    transform_features,
    validate_binary_columns,
)
from views import build_engineered_view, build_linear_view, engineered_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only year-prediction feature views.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("parquets/year_prediction/dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("parquets/year_prediction/features"),
    )
    parser.add_argument("--feature-version", default="v1")
    parser.add_argument("--quantile-error", type=float, default=DEFAULT_QUANTILE_ERROR)
    parser.add_argument("--variance-threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    parser.add_argument("--output-partitions", type=int, default=12)
    return parser.parse_args()


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def split_counts(df: DataFrame) -> dict[str, int]:
    return {
        row[SPLIT]: int(row["count"])
        for row in df.groupBy(SPLIT).count().collect()
    }

