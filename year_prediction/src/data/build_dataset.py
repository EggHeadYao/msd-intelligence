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
    ARTIST_ID,
    AUDIO_FEATURE_COLUMNS,
    AUDIO_INPUT_COLUMNS,
    AUDIO_TYPE_NAMES,
    EXPECTED_INPUT_ROWS,
    EXPECTED_LABELED_ARTISTS,
    EXPECTED_LABELED_TRACKS,
    EXPECTED_OFFICIAL_OMISSIONS,
    EXPECTED_OFFICIAL_TEST_ARTISTS,
    EXPECTED_OFFICIAL_TEST_TRACKS,
    EXPECTED_OFFICIAL_TRAIN_ARTISTS,
    EXPECTED_UNLABELED_TRACKS,
    MAX_YEAR,
    MIN_YEAR,
    OFFICIAL_SPLIT_COMMIT,
    OFFICIAL_TEST_SHA256,
    OFFICIAL_TRAIN_SHA256,
    SPLIT,
    TEST,
    TRACK_ID,
    TRAIN,
    VALIDATION,
    YEAR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the year-prediction dataset contract.")
    parser.add_argument("--metadata", type=Path, default=Path("parquets/prepared/songs_metadata.parquet"))
    parser.add_argument("--audio", type=Path, default=Path("parquets/prepared/song_audio_features_raw.parquet"))
    parser.add_argument("--train-artists", type=Path, default=Path("parquets/year_prediction/source/artists_train.txt"))
    parser.add_argument("--test-artists", type=Path, default=Path("parquets/year_prediction/source/artists_test.txt"))
    parser.add_argument("--output", type=Path, default=Path("parquets/year_prediction/dataset"))
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=472)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    return parser.parse_args()


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and not item.name.startswith(".") and item.name != "_SUCCESS"
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        total_bytes += item.stat().st_size
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {"file_count": len(files), "bytes": total_bytes, "sha256": digest.hexdigest()}


def assert_schema(df: DataFrame, expected: dict[str, str], exact: bool = True) -> None:
    actual = {field.name: field.dataType.simpleString() for field in df.schema.fields}
    if exact and set(actual) != set(expected):
        raise ValueError(f"Schema columns differ: expected={sorted(expected)}, actual={sorted(actual)}")
    wrong = {
        name: (actual.get(name), data_type)
        for name, data_type in expected.items()
        if actual.get(name) != data_type
    }
    if wrong:
        raise ValueError(f"Schema types differ: {wrong}")


