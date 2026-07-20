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


def require_same(left: DataFrame, right: DataFrame, columns: tuple[str, ...], label: str) -> None:
    left_view = left.select(*columns)
    right_view = right.select(*columns)
    require(left_view.exceptAll(right_view).limit(1).count() == 0, f"{label}: unexpected rows")
    require(right_view.exceptAll(left_view).limit(1).count() == 0, f"{label}: missing rows")


def assignment_sha256(assignments: DataFrame) -> str:
    digest = hashlib.sha256()
    rows = assignments.select(ARTIST_ID, SPLIT).distinct().orderBy(ARTIST_ID).toLocalIterator()
    for row in rows:
        digest.update(f"{row[ARTIST_ID]}\t{row[SPLIT]}\n".encode("ascii"))
    return digest.hexdigest()


def split_statistics(labels: DataFrame) -> dict[str, dict[str, int]]:
    rows = labels.groupBy(SPLIT).agg(
        F.count("*").alias("tracks"),
        F.countDistinct(ARTIST_ID).alias("artists"),
    ).collect()
    return {
        row[SPLIT]: {"artists": int(row["artists"]), "tracks": int(row["tracks"])}
        for row in rows
    }


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def validate(args: argparse.Namespace, spark: SparkSession) -> None:
    manifest = load_manifest(args.dataset / "manifest.json")
    require(manifest["contract_version"] == "year_prediction_dataset_v3", "contract version differs")
    require(manifest["format_version"] == 3, "manifest format differs")
    require(manifest["year_contract"] == {
        "minimum": MIN_YEAR,
        "maximum": MAX_YEAR,
        "unlabeled_value": None,
    }, "year contract differs")
    require(sha256_file(args.scalar) == manifest["sources"]["scalar"]["sha256"], "scalar checksum differs")
    require(sha256_file(args.train_artists) == OFFICIAL_TRAIN_SHA256, "train artist checksum differs")
    require(sha256_file(args.test_artists) == OFFICIAL_TEST_SHA256, "test artist checksum differs")
    audio_contract_path = args.audio / "feature_contract.json"
    audio_contract = load_manifest(audio_contract_path)
    require(
        sha256_file(audio_contract_path) == manifest["sources"]["audio_features"]["contract_sha256"],
        "audio contract checksum differs",
    )
    require(audio_contract["contract_version"] == AUDIO_CONTRACT_VERSION, "audio contract version differs")
    require(int(audio_contract["feature_count"]) == AUDIO_FEATURE_COUNT, "audio feature count differs")
    require(len(audio_contract["columns"]) == AUDIO_FEATURE_COUNT + 1, "audio contract columns differ")
    require(audio_contract["columns"][0] == TRACK_ID, "audio contract must start with track_id")
    require(
        audio_contract["feature_order_sha256"] == AUDIO_FEATURE_ORDER_SHA256,
        "audio feature order differs",
    )

    scalar = spark.read.parquet(spark_path(args.scalar))
    labels = spark.read.parquet(spark_path(args.dataset / "labelled_tracks.parquet"))
    assignments = spark.read.parquet(spark_path(args.dataset / "split_assignments.parquet"))
    require_schema(scalar, SCALAR_COLUMNS, SCALAR_TYPES)
    require_schema(labels, LABEL_COLUMNS, LABEL_TYPES)
    require_schema(assignments, ASSIGNMENT_COLUMNS, ASSIGNMENT_TYPES)
    scalar_keys = scalar.select(TRACK_ID, ARTIST_ID, YEAR).persist(StorageLevel.MEMORY_AND_DISK)
    labels.persist(StorageLevel.MEMORY_AND_DISK)
    assignments.persist(StorageLevel.MEMORY_AND_DISK)

    scalar_summary = scalar_keys.agg(
        F.count("*").alias("rows"),
        F.countDistinct(TRACK_ID).alias("tracks"),
        F.sum(F.when(F.col(YEAR).isNull(), 1).otherwise(0)).alias("unlabeled"),
        F.sum(F.when(F.col(YEAR).between(MIN_YEAR, MAX_YEAR), 1).otherwise(0)).alias("labeled"),
        F.sum(
            F.when(F.col(YEAR).isNotNull() & ~F.col(YEAR).between(MIN_YEAR, MAX_YEAR), 1).otherwise(0)
        ).alias("invalid_years"),
    ).first()
    require(int(scalar_summary["rows"]) == EXPECTED_INPUT_TRACKS, "scalar row count differs")
    require(int(scalar_summary["tracks"]) == EXPECTED_INPUT_TRACKS, "scalar track IDs differ")
    require(int(scalar_summary["unlabeled"]) == EXPECTED_UNLABELED_TRACKS, "unlabeled count differs")
    require(int(scalar_summary["labeled"]) == EXPECTED_LABELED_TRACKS, "labeled count differs")
    require(int(scalar_summary["invalid_years"]) == 0, "invalid years found")

    for frame, label in ((labels, "labels"), (assignments, "assignments")):
        summary = frame.agg(
            F.count("*").alias("rows"),
            F.countDistinct(TRACK_ID).alias("tracks"),
            F.sum(
                F.when(
                    F.col(TRACK_ID).isNull()
                    | (F.col(TRACK_ID) == "")
                    | F.col(ARTIST_ID).isNull()
                    | (F.col(ARTIST_ID) == "")
                    | ~F.col(SPLIT).isin(TRAIN, VALIDATION, TEST),
                    1,
                ).otherwise(0)
            ).alias("invalid"),
        ).first()
        require(int(summary["rows"]) == EXPECTED_LABELED_TRACKS, f"{label} row count differs")
        require(int(summary["tracks"]) == EXPECTED_LABELED_TRACKS, f"{label} track IDs differ")
        require(int(summary["invalid"]) == 0, f"{label} contains invalid values")

    official_test = read_artist_ids(spark, args.test_artists)
    labeled_source = scalar_keys.where(F.col(YEAR).isNotNull())
    expected_artists = expected_artist_assignments(
        labeled_source.select(ARTIST_ID).distinct(),
        official_test,
        int(manifest["config"]["split_seed"]),
        int(manifest["config"]["validation_percent"]),
    )
    expected_labels = labeled_source.join(expected_artists, ARTIST_ID).select(*LABEL_COLUMNS)
    expected_assignments = expected_labels.select(*ASSIGNMENT_COLUMNS)
    require_same(labels, expected_labels, LABEL_COLUMNS, "labelled tracks")
    require_same(assignments, expected_assignments, ASSIGNMENT_COLUMNS, "split assignments")
    require_same(labels.drop(YEAR), assignments, ASSIGNMENT_COLUMNS, "label/assignment relation")
    if args.reference_split is not None:
        reference = spark.read.parquet(spark_path(args.reference_split))
        require_same(assignments, reference, ASSIGNMENT_COLUMNS, "reference split")

    stats = split_statistics(labels)
    require(stats == EXPECTED_SPLITS, f"split statistics differ: {stats}")
    require(stats == manifest["counts"]["splits"], "manifest split statistics differ")
    leaks = (
        assignments.select(ARTIST_ID, SPLIT)
        .distinct()
        .groupBy(ARTIST_ID)
        .agg(F.countDistinct(SPLIT).alias("splits"))
        .where(F.col("splits") > 1)
        .limit(1)
        .count()
    )
    require(leaks == 0, "artists leak across splits")
    test_artists = assignments.where(F.col(SPLIT) == TEST).select(ARTIST_ID).distinct()
    require(test_artists.count() == EXPECTED_OFFICIAL_TEST_ARTISTS, "test artist count differs")
    require_same(test_artists, official_test, (ARTIST_ID,), "official test artists")
    require(
        assignment_sha256(assignments) == manifest["artist_assignment_sha256"],
        "artist assignment checksum differs",
    )

    feature_paths = audio_paths(args.audio)
    require(len(feature_paths) == 100, "audio feature batch count differs")
    audio_ids = spark.read.parquet(*feature_paths).select(TRACK_ID)
    audio_summary = audio_ids.agg(
        F.count("*").alias("rows"),
        F.countDistinct(TRACK_ID).alias("tracks"),
    ).first()
    require(int(audio_summary["rows"]) == EXPECTED_INPUT_TRACKS, "audio row count differs")
    require(int(audio_summary["tracks"]) == EXPECTED_INPUT_TRACKS, "audio track IDs differ")
    require_same(audio_ids, scalar_keys, (TRACK_ID,), "audio/scalar coverage")

    require(manifest["counts"]["input_tracks"] == EXPECTED_INPUT_TRACKS, "manifest input count differs")
    require(manifest["counts"]["labeled_tracks"] == EXPECTED_LABELED_TRACKS, "manifest label count differs")
    require(manifest["counts"]["unlabeled_tracks"] == EXPECTED_UNLABELED_TRACKS, "manifest unlabeled count differs")
    require(manifest["counts"]["labeled_artists"] == EXPECTED_LABELED_ARTISTS, "manifest artist count differs")
    require(manifest["schema"]["labelled_tracks"] == LABEL_TYPES, "manifest label schema differs")
    require(manifest["schema"]["split_assignments"] == ASSIGNMENT_TYPES, "manifest split schema differs")

    assignments.unpersist()
    labels.unpersist()
    scalar_keys.unpersist()
    print(
        "year_dataset_valid "
        f"tracks={EXPECTED_INPUT_TRACKS}, labeled={EXPECTED_LABELED_TRACKS}, "
        f"train={stats[TRAIN]['tracks']}, validation={stats[VALIDATION]['tracks']}, "
        f"test={stats[TEST]['tracks']}"
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
        validate(args, spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
