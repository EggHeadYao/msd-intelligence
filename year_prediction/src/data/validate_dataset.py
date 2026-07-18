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
    EXPECTED_LABELED_ARTISTS,
    EXPECTED_LABELED_TRACKS,
    EXPECTED_OFFICIAL_OMISSIONS,
    EXPECTED_OFFICIAL_TEST_ARTISTS,
    EXPECTED_OFFICIAL_TEST_TRACKS,
    EXPECTED_OFFICIAL_TRAIN_ARTISTS,
    MAX_YEAR,
    MIN_YEAR,
    SPLIT,
    SPLIT_VALUES,
    SUPERVISED_COLUMNS,
    TEST,
    TRACK_ID,
    TRAIN,
    VALIDATION,
    YEAR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the year-prediction dataset contract.")
    parser.add_argument("--dataset", type=Path, default=Path("parquets/year_prediction/dataset"))
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
        digest.update(item.relative_to(path).as_posix().encode("utf-8") + b"\0")
        total_bytes += item.stat().st_size
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {"file_count": len(files), "bytes": total_bytes, "sha256": digest.hexdigest()}


def read_artist_ids(spark: SparkSession, path: Path) -> DataFrame:
    return (
        spark.read.text(spark_path(path))
        .select(F.trim(F.col("value")).alias(ARTIST_ID))
        .where(F.col(ARTIST_ID) != "")
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def assignment_sha256(assignments: DataFrame) -> str:
    digest = hashlib.sha256()
    artists = assignments.select(ARTIST_ID, SPLIT).distinct().orderBy(ARTIST_ID).toLocalIterator()
    for row in artists:
        digest.update(f"{row[ARTIST_ID]}\t{row[SPLIT]}\n".encode("ascii"))
    return digest.hexdigest()


def split_statistics(supervised: DataFrame) -> dict[str, dict[str, int]]:
    rows = supervised.groupBy(SPLIT).agg(
        F.count("*").alias("tracks"),
        F.countDistinct(ARTIST_ID).alias("artists"),
    ).collect()
    return {
        row[SPLIT]: {"tracks": int(row["tracks"]), "artists": int(row["artists"])}
        for row in rows
    }


def content_digests(df: DataFrame) -> DataFrame:
    value_columns = (ARTIST_ID, YEAR, *AUDIO_FEATURE_COLUMNS)
    serialized = F.to_json(
        F.struct(*(F.col(column) for column in value_columns)),
        {"ignoreNullFields": "false"},
    )
    return df.select(TRACK_ID, F.sha2(serialized, 256).alias("digest"))


def validate(dataset: Path, spark: SparkSession) -> None:
    manifest_path = dataset / "manifest.json"
    with manifest_path.open("r", encoding="ascii") as handle:
        manifest = json.load(handle)

    supervised = spark.read.parquet(spark_path(dataset / "supervised_features.parquet"))
    assignments = spark.read.parquet(spark_path(dataset / "split_assignments.parquet"))
    supervised.persist(StorageLevel.MEMORY_AND_DISK)
    assignments.persist(StorageLevel.MEMORY_AND_DISK)

    require(tuple(supervised.columns) == SUPERVISED_COLUMNS, f"Unexpected supervised columns: {supervised.columns}")
    supervised_types = {field.name: field.dataType.simpleString() for field in supervised.schema.fields}
    expected_types = {
        TRACK_ID: "string",
        ARTIST_ID: "string",
        YEAR: "int",
        **AUDIO_TYPE_NAMES,
        SPLIT: "string",
    }
    require(supervised_types == expected_types, f"Unexpected supervised schema: {supervised_types}")
    require(
        set(assignments.columns) == {TRACK_ID, ARTIST_ID, SPLIT},
        f"Unexpected assignment columns: {assignments.columns}",
    )

    valid_year = F.col(YEAR).between(MIN_YEAR, MAX_YEAR)
    stats = supervised.agg(
        F.count("*").alias("rows"),
        F.countDistinct(TRACK_ID).alias("tracks"),
        F.countDistinct(ARTIST_ID).alias("artists"),
        F.sum(F.when(~valid_year | F.col(YEAR).isNull(), 1).otherwise(0)).alias("invalid_years"),
        F.sum(
            F.when(
                F.col(TRACK_ID).isNull()
                | (F.col(TRACK_ID) == "")
                | F.col(ARTIST_ID).isNull()
                | (F.col(ARTIST_ID) == "")
                | F.col(SPLIT).isNull(),
                1,
            ).otherwise(0)
        ).alias("invalid_required"),
    ).first()
    require(int(stats["rows"]) == EXPECTED_LABELED_TRACKS, "Wrong supervised row count")
    require(int(stats["tracks"]) == EXPECTED_LABELED_TRACKS, "Duplicate supervised track IDs")
    require(int(stats["artists"]) == EXPECTED_LABELED_ARTISTS, "Wrong supervised artist count")
    require(int(stats["invalid_years"]) == 0, "Invalid supervised years")
    require(int(stats["invalid_required"]) == 0, "Null or empty required values")

    split_values = {row[SPLIT] for row in supervised.select(SPLIT).distinct().collect()}
    require(split_values == set(SPLIT_VALUES), f"Unexpected split values: {split_values}")
    require(
        supervised.groupBy(ARTIST_ID).agg(F.countDistinct(SPLIT).alias("n")).where(F.col("n") != 1).count() == 0,
        "An artist occurs in more than one split",
    )



def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("YearPredictionValidateDataset")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        validate(args.dataset.resolve(), spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
