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
    EXPECTED_OFFICIAL_OMISSIONS,
    EXPECTED_OFFICIAL_TEST_ARTISTS,
    EXPECTED_OFFICIAL_TRAIN_ARTISTS,
    EXPECTED_SPLITS,
    EXPECTED_UNLABELED_TRACKS,
    LABEL_COLUMNS,
    LABEL_TYPES,
    MAX_YEAR,
    MIN_YEAR,
    OFFICIAL_SPLIT_COMMIT,
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
    parser = argparse.ArgumentParser(description="Build the year-prediction dataset contract.")
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
    parser.add_argument("--output", type=Path, default=Path("parquets/year_prediction/dataset"))
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=472)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    return parser.parse_args()


def spark_path(path: str | Path) -> str:
    text = str(path)
    return text if "://" in text else Path(text).resolve().as_uri()


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


def assign_artists(
    labeled_artists: DataFrame,
    official_test: DataFrame,
    split_seed: int,
    validation_percent: int,
) -> DataFrame:
    test_marker = official_test.select(ARTIST_ID).withColumn("_official_test", F.lit(True))
    hash_key = F.concat_ws(":", F.lit(str(split_seed)), F.col(ARTIST_ID))
    return (
        labeled_artists.join(test_marker, ARTIST_ID, "left")
        .withColumn(
            SPLIT,
            F.when(F.col("_official_test").isNotNull(), F.lit(TEST))
            .when(
                F.pmod(F.xxhash64(hash_key), F.lit(10_000)) < validation_percent * 100,
                F.lit(VALIDATION),
            )
            .otherwise(F.lit(TRAIN)),
        )
        .select(ARTIST_ID, SPLIT)
    )


def assignment_sha256(assignments: DataFrame) -> str:
    digest = hashlib.sha256()
    for row in assignments.orderBy(ARTIST_ID).toLocalIterator():
        digest.update(f"{row[ARTIST_ID]}\t{row[SPLIT]}\n".encode("ascii"))
    return digest.hexdigest()


def split_statistics(labeled: DataFrame) -> dict[str, dict[str, int]]:
    rows = labeled.groupBy(SPLIT).agg(
        F.count("*").alias("tracks"),
        F.countDistinct(ARTIST_ID).alias("artists"),
    ).collect()
    return {
        row[SPLIT]: {"artists": int(row["artists"]), "tracks": int(row["tracks"])}
        for row in rows
    }


def source_path(path: Path) -> str:
    return path.as_posix()


def audio_contract(path: Path) -> dict[str, Any]:
    contract_path = path / "feature_contract.json"
    with contract_path.open("r", encoding="ascii") as handle:
        contract = json.load(handle)
    batches = sorted(path.glob("features_*.parquet"))
    require(len(batches) == 100, f"Expected 100 audio feature batches, got {len(batches)}")
    require(contract["contract_version"] == AUDIO_CONTRACT_VERSION, "audio contract version differs")
    require(int(contract["feature_count"]) == AUDIO_FEATURE_COUNT, "audio feature count differs")
    require(len(contract["columns"]) == AUDIO_FEATURE_COUNT + 1, "audio contract columns differ")
    require(contract["columns"][0] == TRACK_ID, "audio contract must start with track_id")
    require(
        contract["feature_order_sha256"] == AUDIO_FEATURE_ORDER_SHA256,
        "audio feature order differs",
    )
    return {
        "path": source_path(path),
        "batch_count": len(batches),
        "bytes": sum(item.stat().st_size for item in batches),
        "contract_path": source_path(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_version": contract["contract_version"],
        "feature_count": int(contract["feature_count"]),
        "feature_order_sha256": contract["feature_order_sha256"],
    }


def build(args: argparse.Namespace, spark: SparkSession) -> Path:
    require(0 < args.validation_percent < 100, "validation percent must be between 1 and 99")
    require(not args.output.exists(), f"Output already exists: {args.output}")
    require(sha256_file(args.train_artists) == OFFICIAL_TRAIN_SHA256, "train artist checksum differs")
    require(sha256_file(args.test_artists) == OFFICIAL_TEST_SHA256, "test artist checksum differs")

    scalar = spark.read.parquet(spark_path(args.scalar))
    require_schema(scalar, SCALAR_COLUMNS, SCALAR_TYPES)
    scalar_keys = scalar.select(TRACK_ID, ARTIST_ID, YEAR).persist(StorageLevel.MEMORY_AND_DISK)
    scalar_row = scalar_keys.agg(
        F.count("*").alias("rows"),
        F.countDistinct(TRACK_ID).alias("tracks"),
        F.sum(F.when(F.col(YEAR).isNull(), 1).otherwise(0)).alias("unlabeled"),
        F.sum(F.when(F.col(YEAR).between(MIN_YEAR, MAX_YEAR), 1).otherwise(0)).alias("labeled"),
        F.sum(
            F.when(F.col(YEAR).isNotNull() & ~F.col(YEAR).between(MIN_YEAR, MAX_YEAR), 1).otherwise(0)
        ).alias("invalid_years"),
        F.sum(
            F.when(
                F.col(TRACK_ID).isNull()
                | (F.col(TRACK_ID) == "")
                | F.col(ARTIST_ID).isNull()
                | (F.col(ARTIST_ID) == ""),
                1,
            ).otherwise(0)
        ).alias("invalid_ids"),
    ).first()
    require(int(scalar_row["rows"]) == EXPECTED_INPUT_TRACKS, "scalar row count differs")
    require(int(scalar_row["tracks"]) == EXPECTED_INPUT_TRACKS, "scalar track IDs are duplicated")
    require(int(scalar_row["unlabeled"]) == EXPECTED_UNLABELED_TRACKS, "unlabeled count differs")
    require(int(scalar_row["labeled"]) == EXPECTED_LABELED_TRACKS, "labeled count differs")
    require(int(scalar_row["invalid_years"]) == 0, "scalar contains invalid years")
    require(int(scalar_row["invalid_ids"]) == 0, "scalar contains invalid IDs")

    labeled_source = scalar_keys.where(F.col(YEAR).isNotNull())
    labeled_artists = labeled_source.select(ARTIST_ID).distinct().persist(StorageLevel.MEMORY_AND_DISK)
    require(labeled_artists.count() == EXPECTED_LABELED_ARTISTS, "labeled artist count differs")
    official_train = read_artist_ids(spark, args.train_artists).persist(StorageLevel.MEMORY_AND_DISK)
    official_test = read_artist_ids(spark, args.test_artists).persist(StorageLevel.MEMORY_AND_DISK)
    require(official_train.count() == EXPECTED_OFFICIAL_TRAIN_ARTISTS, "official train count differs")
    require(official_test.count() == EXPECTED_OFFICIAL_TEST_ARTISTS, "official test count differs")
    require(official_train.distinct().count() == EXPECTED_OFFICIAL_TRAIN_ARTISTS, "duplicate train artists")
    require(official_test.distinct().count() == EXPECTED_OFFICIAL_TEST_ARTISTS, "duplicate test artists")
    require(official_train.join(official_test, ARTIST_ID, "inner").limit(1).count() == 0, "official split overlaps")
    official_union = official_train.union(official_test).distinct()
    require(official_union.join(labeled_artists, ARTIST_ID, "left_anti").count() == 0, "unknown official artists")
    omitted = [
        row[ARTIST_ID]
        for row in labeled_artists.join(official_union, ARTIST_ID, "left_anti").orderBy(ARTIST_ID).collect()
    ]
    require(len(omitted) == EXPECTED_OFFICIAL_OMISSIONS, "official omission count differs")

    artist_assignments = assign_artists(
        labeled_artists,
        official_test,
        args.split_seed,
        args.validation_percent,
    ).persist(StorageLevel.MEMORY_AND_DISK)
    labeled = labeled_source.join(artist_assignments, ARTIST_ID).select(*LABEL_COLUMNS).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    assignments = labeled.select(*ASSIGNMENT_COLUMNS)
    stats = split_statistics(labeled)
    require(stats == EXPECTED_SPLITS, f"split statistics differ: {stats}")

    args.output.mkdir(parents=True)
    labeled.repartition(SPLIT).write.partitionBy(SPLIT).parquet(
        spark_path(args.output / "labelled_tracks.parquet")
    )
    assignments.repartition(SPLIT).write.partitionBy(SPLIT).parquet(
        spark_path(args.output / "split_assignments.parquet")
    )
    manifest = {
        "contract_version": "year_prediction_dataset_v3",
        "format_version": 3,
        "year_contract": {"minimum": MIN_YEAR, "maximum": MAX_YEAR, "unlabeled_value": None},
        "config": {
            "split_seed": args.split_seed,
            "validation_percent": args.validation_percent,
            "validation_hash": "pmod(xxhash64(seed + ':' + artist_id), 10000)",
        },
        "sources": {
            "scalar": {
                "path": source_path(args.scalar),
                "sha256": sha256_file(args.scalar),
                "columns": len(SCALAR_COLUMNS),
                "tracks": EXPECTED_INPUT_TRACKS,
            },
            "audio_features": audio_contract(args.audio),
            "official_split_commit": OFFICIAL_SPLIT_COMMIT,
            "official_train": {
                "path": source_path(args.train_artists),
                "sha256": OFFICIAL_TRAIN_SHA256,
            },
            "official_test": {
                "path": source_path(args.test_artists),
                "sha256": OFFICIAL_TEST_SHA256,
            },
        },
        "counts": {
            "input_tracks": EXPECTED_INPUT_TRACKS,
            "labeled_tracks": EXPECTED_LABELED_TRACKS,
            "unlabeled_tracks": EXPECTED_UNLABELED_TRACKS,
            "labeled_artists": EXPECTED_LABELED_ARTISTS,
            "official_split_omitted_artists": omitted,
            "splits": stats,
        },
        "schema": {
            "labelled_tracks": LABEL_TYPES,
            "split_assignments": ASSIGNMENT_TYPES,
        },
        "artist_assignment_sha256": assignment_sha256(artist_assignments),
    }
    with (args.output / "manifest.json").open("w", encoding="ascii") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    labeled.unpersist()
    artist_assignments.unpersist()
    official_test.unpersist()
    official_train.unpersist()
    labeled_artists.unpersist()
    scalar_keys.unpersist()
    return args.output


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("YearPredictionBuildDataset")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        output = build(args, spark)
        print(f"year_dataset_built output={output.resolve()}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
