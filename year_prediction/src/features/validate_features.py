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


def assert_nested_close(actual: Any, expected: Any, path: str = "preprocessing") -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Keys differ at {path}")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Length differs at {path}")
        for index, value in enumerate(expected):
            assert_nested_close(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        require(
            isinstance(actual, (int, float))
            and math.isclose(float(actual), expected, rel_tol=1.0e-10, abs_tol=1.0e-12),
            f"Numeric value differs at {path}: actual={actual}, expected={expected}",
        )
    else:
        require(actual == expected, f"Value differs at {path}: actual={actual}, expected={expected}")


def require_exact_content(actual: DataFrame, expected: DataFrame, columns: list[str], label: str) -> None:
    def digests(df: DataFrame, alias: str) -> DataFrame:
        value = F.to_json(F.struct(*(F.col(column) for column in columns)), {"ignoreNullFields": "false"})
        return df.select(TRACK_ID, F.sha2(value, 256).alias(alias))

    joined = digests(actual, "actual").join(digests(expected, "expected"), TRACK_ID, "full_outer")
    mismatches = joined.where(
        F.col("actual").isNull()
        | F.col("expected").isNull()
        | (F.col("actual") != F.col("expected"))
    ).count()
    require(mismatches == 0, f"{label} content differs from a train-only reconstruction")


def validate(features: Path, dataset: Path, spark: SparkSession) -> None:


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("YearPredictionValidateFeatures")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        validate(args.features.resolve(), args.dataset.resolve(), spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
