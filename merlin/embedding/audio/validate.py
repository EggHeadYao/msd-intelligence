from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType

from shared_contract import CONTRACT_VERSION


EXPECTED_SONGS = 1_000_000
EMBEDDING_COLUMN = "embedding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MERLIN C1 PCA audio outputs.")
    parser.add_argument("--output", type=Path, default=Path("parquets_new/merlin/audio"))
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_SONGS)
    parser.add_argument("--norm-tolerance", type=float, default=1e-6)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinValidateAudioPCA")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_layout(output_dir: Path) -> None:
    required = (
        "song_embeddings_audio.parquet",
        "audio_encoder_metadata.json",
        "pca_model",
        "scaler_model",
    )
    missing = [name for name in required if not (output_dir / name).exists()]
    require(not missing, f"audio output missing files: {missing}")


def validate_metadata(metadata: dict[str, Any]) -> int:
    required = (
        "merlin_schema_version",
        "shared_audio_contract_version",
        "c1_feature_version",
        "model_ready_schema_version",
        "shared_audio_feature_count",
        "merlin_array_feature_count",
        "merlin_raw_view_count",
        "row_count",
        "input_schema_sha256",
        "feature_columns",
        "feature_count",
        "permanent_dropped_fields",
        "feature_order_sha256",
        "embedding_format",
        "selected_k",
        "explained_variance",
        "cumulative_explained_variance",
        "preprocess",
        "scaler_mean",
        "scaler_std",
    )
    missing = [key for key in required if key not in metadata]
    require(not missing, f"metadata missing keys: {missing}")
    selected_k = int(metadata["selected_k"])
    require(metadata["merlin_schema_version"] == "3.0", "wrong MERLIN schema version")
    require(metadata["shared_audio_contract_version"] == CONTRACT_VERSION, "wrong audio contract")
    require(int(metadata["c1_feature_version"]) == 2, "wrong C1 feature version")
    require(metadata["model_ready_schema_version"] == "c1_model_ready_v2", "wrong model schema")
    require(metadata["permanent_dropped_fields"] == ["danceability", "energy"], "wrong dropped fields")
    require(int(metadata["shared_audio_feature_count"]) == 628, "shared feature count mismatch")
    require(int(metadata["merlin_array_feature_count"]) == 552, "array feature count mismatch")
    require(int(metadata["merlin_raw_view_count"]) == 563, "raw view count mismatch")
    require(metadata["embedding_format"] == "array<float32>", "wrong embedding format")
    require(selected_k == 128, "C1 embedding dimension must be 128")
    require(len(metadata["feature_columns"]) == int(metadata["feature_count"]), "feature_count mismatch")
    schema_hash = metadata["input_schema_sha256"]
    require(isinstance(schema_hash, str) and len(schema_hash) == 64, "invalid input schema hash")
    feature_text = "\n".join(metadata["feature_columns"])
    expected_hash = hashlib.sha256(feature_text.encode("utf-8")).hexdigest()
    require(metadata["feature_order_sha256"] == expected_hash, "feature order hash mismatch")
    require(len(metadata["explained_variance"]) >= selected_k, "explained_variance shorter than selected_k")
    require(len(metadata["scaler_mean"]) == int(metadata["feature_count"]), "scaler_mean length mismatch")
    require(len(metadata["scaler_std"]) == int(metadata["feature_count"]), "scaler_std length mismatch")
    return selected_k


def validate_embeddings(
    embeddings: DataFrame,
    expected_rows: int,
    selected_k: int,
    norm_tolerance: float,
) -> None:
    require("track_id" in embeddings.columns, "embeddings missing track_id")
    require(EMBEDDING_COLUMN in embeddings.columns, f"embeddings missing {EMBEDDING_COLUMN}")
    embedding_type = embeddings.schema[EMBEDDING_COLUMN].dataType
    require(
        isinstance(embedding_type, ArrayType) and isinstance(embedding_type.elementType, FloatType),
        "embedding column must be array<float32>",
    )

    row_count = embeddings.count()
    distinct_tracks = embeddings.select("track_id").distinct().count()
    null_rows = embeddings.where(F.col("track_id").isNull() | F.col(EMBEDDING_COLUMN).isNull()).count()
    bad_size = embeddings.where(F.size(F.col(EMBEDDING_COLUMN)) != selected_k).count()
    has_bad_value = F.exists(
        F.col(EMBEDDING_COLUMN),
        lambda x: x.isNull() | F.isnan(x) | (x == float("inf")) | (x == float("-inf")),
    )
    norm = F.sqrt(
        F.aggregate(
            F.col(EMBEDDING_COLUMN),
            F.lit(0.0),
            lambda acc, x: acc + x * x,
        ),
    )
    bad_values = embeddings.where(has_bad_value | (F.abs(norm - F.lit(1.0)) > norm_tolerance)).count()

    print(
        "audio_embeddings "
        f"rows={row_count}, distinct_track_id={distinct_tracks}, "
        f"selected_k={selected_k}, null_rows={null_rows}, "
        f"bad_size={bad_size}, bad_values={bad_values}",
    )
    require(row_count == expected_rows, "audio embedding row count mismatch")
    require(distinct_tracks == expected_rows, "audio embedding track_id mismatch")
    require(null_rows == 0, "audio embeddings contain null rows")
    require(bad_size == 0, "audio embeddings have inconsistent dimensions")
    require(bad_values == 0, "audio embeddings contain NaN/Inf or non-normalized rows")


def main() -> None:
    args = parse_args()
    validate_layout(args.output)
    metadata = read_metadata(args.output / "audio_encoder_metadata.json")
    selected_k = validate_metadata(metadata)
    require(int(metadata["row_count"]) == args.expected_rows, "metadata row_count mismatch")

    spark = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        embeddings = spark.read.parquet(spark_path(args.output / "song_embeddings_audio.parquet"))
        validate_embeddings(embeddings, args.expected_rows, selected_k, args.norm_tolerance)
        print("MERLIN audio PCA validation passed.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
