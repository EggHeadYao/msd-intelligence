from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from columns import (
    AUDIO_CONTRACT_VERSION,
    AUDIO_FEATURE_COUNT,
    AUDIO_FEATURE_ORDER_SHA256,
    ARTIST_ID,
    ASSIGNMENT_COLUMNS,
    ASSIGNMENT_TYPES,
    EXPECTED_INPUT_TRACKS,
    EXPECTED_LABELED_ARTISTS,
    EXPECTED_LABELED_TRACKS,
    EXPECTED_OFFICIAL_TEST_ARTISTS,
    EXPECTED_SPLITS,
    EXPECTED_UNLABELED_TRACKS,
    LABEL_COLUMNS,
    LABEL_TYPES,
    MAX_YEAR,
    MIN_YEAR,
    OFFICIAL_TEST_SHA256,
    OFFICIAL_TRAIN_SHA256,
    SCALAR_COLUMNS,
    SCALAR_TYPES,
    SPLIT,
    TEST,
    TRACK_ID,
    TRAIN,
    VALIDATION,
    YEAR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the year-prediction dataset contract.")
    parser.add_argument("--dataset", type=Path, default=Path("parquets/year_prediction/dataset"))
    parser.add_argument(
        "--scalar",
        type=Path,
        default=Path("parquets/year_prediction/raw/songs_scalar.parquet"),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("parquets/year_prediction/raw/audio_features"),
    )
    parser.add_argument(
        "--train-artists",
        type=Path,
        default=Path("parquets/year_prediction/source/artists_train.txt"),
    )
    parser.add_argument(
        "--test-artists",
        type=Path,
        default=Path("parquets/year_prediction/source/artists_test.txt"),
    )
    parser.add_argument("--reference-split", type=Path)
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


def schema_types(frame: DataFrame) -> dict[str, str]:
    return {field.name: field.dataType.simpleString() for field in frame.schema.fields}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_schema(frame: DataFrame, columns: tuple[str, ...], types: dict[str, str]) -> None:
    require(tuple(frame.columns) == columns, f"Unexpected columns: {frame.columns}")
    require(schema_types(frame) == types, f"Unexpected schema: {schema_types(frame)}")


def read_artist_ids(spark: SparkSession, path: Path) -> DataFrame:
    return (
        spark.read.text(spark_path(path))
        .select(F.trim("value").alias(ARTIST_ID))
        .where(F.col(ARTIST_ID) != "")
    )


def expected_artist_assignments(
    artists: DataFrame,
    official_test: DataFrame,
    seed: int,
    validation_percent: int,
) -> DataFrame:
    marked_test = official_test.select(ARTIST_ID).withColumn("_test", F.lit(True))
    key = F.concat_ws(":", F.lit(str(seed)), F.col(ARTIST_ID))
    return (
        artists.join(marked_test, ARTIST_ID, "left")
        .select(
            ARTIST_ID,
            F.when(F.col("_test").isNotNull(), F.lit(TEST))
            .when(
                F.pmod(F.xxhash64(key), F.lit(10_000)) < validation_percent * 100,
                F.lit(VALIDATION),
            )
            .otherwise(F.lit(TRAIN))
            .alias(SPLIT),
        )
    )


