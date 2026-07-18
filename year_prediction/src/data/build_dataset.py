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


def read_artist_ids(spark: SparkSession, path: Path) -> DataFrame:
    return (
        spark.read.text(spark_path(path))
        .select(F.trim(F.col("value")).alias(ARTIST_ID))
        .where(F.col(ARTIST_ID) != "")
    )


def assignment_sha256(assignments: DataFrame) -> str:
    digest = hashlib.sha256()
    rows = assignments.select(ARTIST_ID, SPLIT).orderBy(ARTIST_ID).toLocalIterator()
    for row in rows:
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


def require_equal(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected}, got {actual}")


def build(args: argparse.Namespace, spark: SparkSession) -> None:
    if not 0 < args.validation_percent < 100:
        raise ValueError("--validation-percent must be between 1 and 99")
    require_equal(int(sha256_file(args.train_artists) == OFFICIAL_TRAIN_SHA256), 1, "train split checksum")
    require_equal(int(sha256_file(args.test_artists) == OFFICIAL_TEST_SHA256), 1, "test split checksum")

    metadata = spark.read.parquet(spark_path(args.metadata)).select(TRACK_ID, ARTIST_ID, YEAR)
    audio_raw = spark.read.parquet(spark_path(args.audio))
    assert_schema(audio_raw, {TRACK_ID: "string", **AUDIO_TYPE_NAMES})
    audio = audio_raw.select(*AUDIO_INPUT_COLUMNS)
    assert_schema(metadata, {TRACK_ID: "string", ARTIST_ID: "string", YEAR: "int"})
    metadata.persist(StorageLevel.MEMORY_AND_DISK)
    audio.persist(StorageLevel.MEMORY_AND_DISK)

    valid_year = F.col(YEAR).between(MIN_YEAR, MAX_YEAR)
    invalid_year = F.col(YEAR).isNull() | (F.col(YEAR) < 0) | ((F.col(YEAR) > 0) & ~valid_year)
    metadata_stats = metadata.agg(
        F.count("*").alias("rows"),
        F.countDistinct(TRACK_ID).alias("tracks"),
        F.sum(F.when(F.col(TRACK_ID).isNull() | (F.col(TRACK_ID) == ""), 1).otherwise(0)).alias("bad_track_ids"),
        F.sum(F.when(F.col(YEAR) == 0, 1).otherwise(0)).alias("unlabeled"),
        F.sum(F.when(valid_year, 1).otherwise(0)).alias("labeled"),
        F.sum(F.when(invalid_year, 1).otherwise(0)).alias("invalid_years"),
        F.countDistinct(F.when(valid_year, F.col(ARTIST_ID))).alias("labeled_artists"),
    ).first()
    audio_stats = audio.agg(
        F.count("*").alias("rows"),
        F.countDistinct(TRACK_ID).alias("tracks"),
        F.sum(F.when(F.col(TRACK_ID).isNull() | (F.col(TRACK_ID) == ""), 1).otherwise(0)).alias("bad_track_ids"),
    ).first()
    for row, label in ((metadata_stats, "metadata"), (audio_stats, "audio")):
        require_equal(int(row["rows"]), EXPECTED_INPUT_ROWS, f"{label} rows")
        require_equal(int(row["tracks"]), EXPECTED_INPUT_ROWS, f"{label} unique tracks")
        require_equal(int(row["bad_track_ids"]), 0, f"{label} invalid track IDs")
    require_equal(int(metadata_stats["unlabeled"]), EXPECTED_UNLABELED_TRACKS, "unlabeled tracks")
    require_equal(int(metadata_stats["labeled"]), EXPECTED_LABELED_TRACKS, "labeled tracks")
    require_equal(int(metadata_stats["invalid_years"]), 0, "invalid years")
    require_equal(int(metadata_stats["labeled_artists"]), EXPECTED_LABELED_ARTISTS, "labeled artists")
    input_intersection = metadata.select(TRACK_ID).join(
        audio.select(TRACK_ID), TRACK_ID, "inner"
    ).count()
    require_equal(input_intersection, EXPECTED_INPUT_ROWS, "input track intersection")

    labeled_metadata = metadata.where(valid_year).select(TRACK_ID, ARTIST_ID, YEAR)
    bad_artists = labeled_metadata.where(F.col(ARTIST_ID).isNull() | (F.col(ARTIST_ID) == "")).count()
    require_equal(bad_artists, 0, "invalid labeled artist IDs")
    labeled_artists = labeled_metadata.select(ARTIST_ID).distinct().persist(StorageLevel.MEMORY_AND_DISK)

    official_train = read_artist_ids(spark, args.train_artists).persist(StorageLevel.MEMORY_AND_DISK)
    official_test = read_artist_ids(spark, args.test_artists).persist(StorageLevel.MEMORY_AND_DISK)
    require_equal(official_train.count(), EXPECTED_OFFICIAL_TRAIN_ARTISTS, "official train artists")
    require_equal(official_train.distinct().count(), EXPECTED_OFFICIAL_TRAIN_ARTISTS, "distinct official train artists")
    require_equal(official_test.count(), EXPECTED_OFFICIAL_TEST_ARTISTS, "official test artists")
    require_equal(official_test.distinct().count(), EXPECTED_OFFICIAL_TEST_ARTISTS, "distinct official test artists")
    require_equal(official_train.join(official_test, ARTIST_ID, "inner").count(), 0, "official split overlap")

    official_union = official_train.union(official_test).distinct()
    unknown_official = official_union.join(
        labeled_artists, ARTIST_ID, "left_anti"
    ).count()
    require_equal(unknown_official, 0, "official artists absent from labeled data")
    omitted_rows = labeled_artists.join(
        official_union, ARTIST_ID, "left_anti"
    ).orderBy(ARTIST_ID).collect()
    omitted = [row[ARTIST_ID] for row in omitted_rows]
    require_equal(len(omitted), EXPECTED_OFFICIAL_OMISSIONS, "labeled artists omitted by official files")

    test_marker = official_test.select(ARTIST_ID).withColumn("_official_test", F.lit(True))
    hash_key = F.concat_ws(":", F.lit(str(args.split_seed)), F.col(ARTIST_ID))
    validation_bucket_limit = args.validation_percent * 100
    artist_assignments = (
        labeled_artists.join(test_marker, ARTIST_ID, "left")
        .withColumn(
            SPLIT,
            F.when(F.col("_official_test"), F.lit(TEST))
            .when(F.pmod(F.xxhash64(hash_key), F.lit(10_000)) < validation_bucket_limit, F.lit(VALIDATION))
            .otherwise(F.lit(TRAIN)),
        )
        .select(ARTIST_ID, SPLIT)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    supervised = (
        labeled_metadata.join(audio, TRACK_ID, "inner")
        .join(artist_assignments, ARTIST_ID, "inner")
        .select(TRACK_ID, ARTIST_ID, YEAR, *AUDIO_FEATURE_COLUMNS, SPLIT)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    require_equal(supervised.count(), EXPECTED_LABELED_TRACKS, "supervised rows")
    stats = split_statistics(supervised)
    require_equal(stats[TEST]["tracks"], EXPECTED_OFFICIAL_TEST_TRACKS, "official test tracks")
    require_equal(stats[TEST]["artists"], EXPECTED_OFFICIAL_TEST_ARTISTS, "official test artists in output")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    supervised.repartition(SPLIT).write.mode("overwrite").partitionBy(SPLIT).parquet(
        spark_path(output / "supervised_features.parquet")
    )
    supervised.select(TRACK_ID, ARTIST_ID, SPLIT).repartition(SPLIT).write.mode("overwrite").partitionBy(SPLIT).parquet(
        spark_path(output / "split_assignments.parquet")
    )

    train_sha = sha256_file(args.train_artists)
    test_sha = sha256_file(args.test_artists)
    manifest = {
        "format_version": 1,
        "year_contract": {"minimum": MIN_YEAR, "maximum": MAX_YEAR, "unlabeled_value": 0},
        "config": {
            "split_seed": args.split_seed,
            "validation_percent": args.validation_percent,
            "validation_hash": "pmod(xxhash64(seed + ':' + artist_id), 10000)",
        },
        "sources": {
            "metadata": {"path": str(args.metadata.resolve()), **parquet_snapshot(args.metadata)},
            "audio": {"path": str(args.audio.resolve()), **parquet_snapshot(args.audio)},
            "official_split_commit": OFFICIAL_SPLIT_COMMIT,
            "official_train": {"path": str(args.train_artists.resolve()), "sha256": train_sha},
            "official_test": {"path": str(args.test_artists.resolve()), "sha256": test_sha},
        },
        "counts": {
            "input_tracks": EXPECTED_INPUT_ROWS,
            "labeled_tracks": EXPECTED_LABELED_TRACKS,
            "unlabeled_tracks": EXPECTED_UNLABELED_TRACKS,
            "labeled_artists": EXPECTED_LABELED_ARTISTS,
            "official_split_omitted_artists": omitted,
            "splits": stats,
        },
        "schema": {
            "identifiers": {TRACK_ID: "string", ARTIST_ID: "string"},
            "label": {YEAR: "int"},
            "predictors": AUDIO_TYPE_NAMES,
            "partition": {SPLIT: "string"},
        },
        "artist_assignment_sha256": assignment_sha256(artist_assignments),
    }
    with (output / "manifest.json").open("w", encoding="ascii") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")



def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("YearPredictionBuildDataset")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        build(args, spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
