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


def build_metadata(scalar: DataFrame, labels: DataFrame) -> DataFrame:
    labeled = labels.select(
        TRACK_ID,
        F.col(ARTIST_ID).alias("_label_artist_id"),
        F.col(YEAR).alias("_label_year"),
        F.col(SPLIT).alias("_label_split"),
    )
    joined = scalar.select(
        TRACK_ID,
        ARTIST_ID,
        YEAR,
        *GLOBAL_SCALAR_COLUMNS,
    ).join(labeled, TRACK_ID, "left")
    label_present = F.col("_label_split").isNotNull()
    invalid = (
        (F.col(YEAR).isNotNull() != label_present)
        | (
            label_present
            & (
                ~F.col(ARTIST_ID).eqNullSafe(F.col("_label_artist_id"))
                | ~F.col(YEAR).eqNullSafe(F.col("_label_year"))
            )
        )
    )
    summary = joined.agg(
        F.count("*").alias("rows"),
        F.countDistinct(TRACK_ID).alias("tracks"),
        F.sum(F.when(label_present, 1).otherwise(0)).alias("labeled"),
        F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid"),
    ).first()
    require(int(summary["rows"]) == EXPECTED_TRACKS, "metadata row count differs")
    require(int(summary["tracks"]) == EXPECTED_TRACKS, "metadata track IDs are duplicated")
    require(int(summary["labeled"]) == EXPECTED_LABELED_TRACKS, "metadata label count differs")
    require(int(summary["invalid"]) == 0, "dataset labels disagree with scalar data")
    return joined.select(
        TRACK_ID,
        ARTIST_ID,
        YEAR,
        F.col("_label_split").alias(SPLIT),
        *GLOBAL_SCALAR_COLUMNS,
    )


def predictor_metadata(columns: tuple[str, ...], types: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "name": column,
            "type": types[column],
            "source": column_source(column),
            "unit": column_unit(column),
            "missing": column_missing_rule(column),
        }
        for column in columns
    ]


def build(args: argparse.Namespace, spark: SparkSession) -> Path:
    require(not args.output.exists(), f"Output already exists: {args.output}")
    paths = audio_paths(args.audio)
    require(len(paths) == 100, f"Expected 100 audio batches, got {len(paths)}")

    audio_contract_path = args.audio / "feature_contract.json"
    audio_contract = load_audio_contract(audio_contract_path)
    dataset_manifest_path = args.dataset / "manifest.json"
    dataset_manifest = load_json(dataset_manifest_path)
    scalar_sha256 = sha256_file(args.scalar)
    audio_contract_sha256 = sha256_file(audio_contract_path)
    require(dataset_manifest["contract_version"] == "year_prediction_dataset_v3", "dataset contract differs")
    require(dataset_manifest["sources"]["scalar"]["sha256"] == scalar_sha256, "scalar checksum differs")
    require(
        dataset_manifest["sources"]["audio_features"]["contract_sha256"]
        == audio_contract_sha256,
        "audio contract checksum differs",
    )

    audio = spark.read.parquet(*paths)
    scalar = spark.read.parquet(spark_path(args.scalar))
    labels = spark.read.parquet(spark_path(args.dataset / "labelled_tracks.parquet"))
    require(tuple(audio.columns) == tuple(audio_contract["columns"]), "audio schema order differs")
    require_types(
        audio,
        {TRACK_ID: "string", **{column: "double" for column in audio.columns[1:]}},
        "audio",
    )
    require(tuple(labels.columns) == LABEL_COLUMNS, "label schema order differs")
    require_types(labels, LABEL_TYPES, "labels")
    require_types(scalar, SCALAR_TYPES, "scalar")

    audio_summary = audio.agg(
        F.count("*").alias("rows"),
        F.countDistinct(TRACK_ID).alias("tracks"),
        F.sum(F.when(F.col(TRACK_ID).isNull() | (F.col(TRACK_ID) == ""), 1).otherwise(0)).alias(
            "invalid"
        ),
    ).first()
    require(int(audio_summary["rows"]) == EXPECTED_TRACKS, "audio row count differs")
    require(int(audio_summary["tracks"]) == EXPECTED_TRACKS, "audio track IDs are duplicated")
    require(int(audio_summary["invalid"]) == 0, "audio contains invalid track IDs")

    metadata = build_metadata(scalar, labels).persist(StorageLevel.DISK_ONLY)
    metadata.count()
    require(
        audio.select(TRACK_ID).join(metadata.select(TRACK_ID), TRACK_ID, "left_anti").limit(1).count()
        == 0,
        "audio and metadata track coverage differs",
    )

    shared_columns = year_shared_columns(audio_contract)
    joined = audio.join(F.broadcast(metadata), TRACK_ID, "inner")
    t90 = build_t90(joined)
    full = build_full_tabular(joined, shared_columns)
    require(len(t90.columns) == len(AUDIT_COLUMNS) + 90, "T90 dimension differs")
    require(len(full.columns) == len(AUDIT_COLUMNS) + 594, "full dimension differs")
    require(
        not (set(full.columns) & set(FORBIDDEN_PREDICTOR_COLUMNS)) - set(AUDIT_COLUMNS),
        "forbidden predictors found",
    )

    full.write.parquet(spark_path(args.output / "full_tabular.parquet"))
    t90.write.parquet(spark_path(args.output / "t90.parquet"))
    full_types = schema_types(full)
    predictor_columns = full_predictor_columns(audio_contract)
    manifest = {
        "contract_version": FEATURE_CONTRACT_VERSION,
        "format_version": 1,
        "sources": {
            "audio": {
                "path": args.audio.as_posix(),
                "contract_sha256": audio_contract_sha256,
                "contract_version": audio_contract["contract_version"],
                "feature_count": AUDIO_FEATURE_COUNT,
                "feature_order_sha256": audio_contract["feature_order_sha256"],
            },
            "scalar": {"path": args.scalar.as_posix(), "sha256": scalar_sha256},
            "dataset": {
                "path": args.dataset.as_posix(),
                "manifest_sha256": sha256_file(dataset_manifest_path),
                "contract_version": dataset_manifest["contract_version"],
            },
        },
        "counts": {
            "tracks": EXPECTED_TRACKS,
            "labeled_tracks": EXPECTED_LABELED_TRACKS,
            "unlabeled_tracks": EXPECTED_TRACKS - EXPECTED_LABELED_TRACKS,
            "splits": dataset_manifest["counts"]["splits"],
        },
        "audit_columns": list(AUDIT_COLUMNS),
        "views": {
            "t90": {
                "path": "t90.parquet",
                "predictor_count": len(T90_COLUMNS),
                "predictor_columns": list(T90_COLUMNS),
                "predictor_order_sha256": order_sha256(T90_COLUMNS),
                "schema": schema_payload(t90),
            },
            "full_tabular": {
                "path": "full_tabular.parquet",
                "shared_predictor_count": len(shared_columns),
                "predictor_count": len(predictor_columns),
                "predictor_columns": list(predictor_columns),
                "predictor_order_sha256": order_sha256(predictor_columns),
                "schema": schema_payload(full),
            },
        },
        "feature_groups": {
            name: list(columns) for name, columns in ordered_feature_groups(audio_contract).items()
        },
        "excluded_audio_columns": list(
            column for column in audio.columns[1:] if column not in shared_columns
        ),
        "derived_features": {
            "fade_in_ratio": "clip(end_of_fade_in, 0, duration) / duration",
            "fade_out_ratio": "(duration - clip(start_of_fade_out, 0, duration)) / duration",
            "active_audio_ratio": "(clip(start_of_fade_out) - clip(end_of_fade_in)) / duration",
            "endpoint_tolerance_seconds": FADE_TOLERANCE_SECONDS,
        },
        "predictor_contract": predictor_metadata(predictor_columns, full_types),
        "forbidden_predictor_columns": list(FORBIDDEN_PREDICTOR_COLUMNS),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.json").open("w", encoding="ascii") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    metadata.unpersist()
    return args.output


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("YearPredictionBuildFeatures")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.sql.debug.maxToStringFields", "1000")
        .config("spark.sql.broadcastTimeout", "1200")
        .config("spark.hadoop.parquet.block.size", str(16 * 1024 * 1024))
        .config("spark.hadoop.parquet.page.size", str(64 * 1024))
        .config("spark.hadoop.parquet.dictionary.page.size", str(64 * 1024))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        output = build(args, spark)
        print(f"year_features_built output={output.resolve()}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
