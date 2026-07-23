from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any, Sequence

from pyspark import StorageLevel
from pyspark.ml.feature import PCAModel, StandardScalerModel, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType

from artifacts import C1_MANIFEST_NAME, sha256_path, validate_c1_manifest
from columns import (
    CONTRACT_VERSION,
    PREPARED_AUDIO_COLUMNS,
    TRACK_ID_COLUMN,
    TIME_SIGNATURE_UNKNOWN_COLUMN,
    TIME_SIGNATURE_VALUES,
    build_feature_columns,
    time_signature_one_hot_column,
)
from l1_stats import (
    bootstrap_hedges_g_ci,
    classify_validation,
    distribution,
    hedges_g,
    preservation_summary,
)
from preprocess import (
    SEGMENT_MEDIAN_BATCH_SIZE,
    add_scalar_availability,
    apply_frozen_preprocess,
)
from train_pca import (
    FEATURES_COLUMN,
    PCA_FEATURES_COLUMN,
    SCALED_FEATURES_COLUMN,
    add_normalized_embedding,
)


EXPECTED_SONGS = 1_000_000
EMBEDDING_COLUMN = "embedding"
VALIDATION_VERSION = "c1_l1_1_v1"
PAIR_TYPES = ("same_artist", "same_release", "random")
METADATA_COLUMNS = (
    TRACK_ID_COLUMN,
    "song_id",
    "artist_id",
    "release_7digitalid",
    "year",
    "has_year",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MERLIN C1 audio artifacts and L1-1.")
    parser.add_argument("--mode", choices=("artifact", "l1", "all"), default="artifact")
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=Path("parquets_new/prepared/song_audio_features_raw.parquet"),
    )
    parser.add_argument(
        "--songs-metadata",
        type=Path,
        default=Path("parquets_new/prepared/songs_metadata.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("parquets_new/merlin/audio"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_SONGS)
    parser.add_argument("--norm-tolerance", type=float, default=1e-6)
    parser.add_argument("--pair-count", type=int, default=10_000)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--reproduction-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--allow-partial-pairs",
        action="store_true",
        help="Smoke-only mode: report fewer than the required pairs instead of failing.",
    )
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinValidateAudioC1")
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
        C1_MANIFEST_NAME,
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
        "run_id",
        "producer",
        "input_path",
        "row_count",
        "input_schema_sha256",
        "parent_prepared_manifest",
        "feature_columns",
        "feature_count",
        "permanent_dropped_fields",
        "feature_order_sha256",
        "embedding_format",
        "selected_k",
        "explained_variance",
        "cumulative_explained_variance",
        "pca_128_below_90_percent",
        "preprocess",
        "scaler_mean",
        "scaler_std",
        "limit",
        "shuffle_partitions",
        "segment_median_batch_size",
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
    producer = metadata["producer"]
    require(isinstance(producer, dict), "invalid C1 producer")
    require(isinstance(producer.get("commit"), str) and len(producer["commit"]) >= 40, "invalid commit")
    require(isinstance(producer.get("dirty"), bool), "invalid dirty flag")
    if int(metadata["limit"]) == 0:
        require(not producer["dirty"], "formal C1 artifact was produced by dirty code")
    require(int(metadata["shuffle_partitions"]) > 0, "invalid shuffle partitions")
    require(
        int(metadata["segment_median_batch_size"]) == SEGMENT_MEDIAN_BATCH_SIZE,
        "segment median batch size mismatch",
    )
    schema_hash = metadata["input_schema_sha256"]
    require(isinstance(schema_hash, str) and len(schema_hash) == 64, "invalid input schema hash")
    parent = metadata["parent_prepared_manifest"]
    require(isinstance(parent, dict), "invalid Prepared parent lineage")
    require(parent.get("artifact_type") == "prepared_tables", "wrong parent artifact type")
    require(parent.get("artifact_version") == "v2", "wrong parent artifact version")
    require(parent.get("shared_audio_contract_version") == CONTRACT_VERSION, "wrong parent contract")
    parent_path = Path(parent.get("path", ""))
    require(parent_path.is_file(), "Prepared parent manifest is unavailable")
    require(parent.get("sha256") == sha256_path(parent_path), "Prepared parent manifest hash mismatch")
    feature_text = "\n".join(metadata["feature_columns"])
    expected_hash = hashlib.sha256(feature_text.encode("utf-8")).hexdigest()
    require(metadata["feature_order_sha256"] == expected_hash, "feature order hash mismatch")
    preprocess = metadata["preprocess"]
    time_values = tuple(preprocess.get("time_signature_values", ()))
    require(time_values == TIME_SIGNATURE_VALUES, "time signature values mismatch")
    time_columns = tuple(time_signature_one_hot_column(value) for value in time_values)
    time_columns = (*time_columns, TIME_SIGNATURE_UNKNOWN_COLUMN)
    require(tuple(preprocess.get("time_signature_columns", ())) == time_columns, "time columns mismatch")
    candidates = build_feature_columns(time_columns)
    dropped = tuple(preprocess.get("dropped_features", ()))
    require(len(dropped) == len(set(dropped)), "duplicate dropped features")
    require(set(dropped).issubset(candidates), "unknown dropped features")
    expected_features = tuple(column for column in candidates if column not in set(dropped))
    require(tuple(metadata["feature_columns"]) == expected_features, "feature schema is not canonical")
    require(len(metadata["explained_variance"]) >= selected_k, "explained_variance shorter than selected_k")
    cumulative_128 = float(metadata["cumulative_explained_variance"][selected_k - 1])
    require(
        bool(metadata["pca_128_below_90_percent"]) == (cumulative_128 < 0.90),
        "PCA-128 variance diagnostic mismatch",
    )
    require(len(metadata["scaler_mean"]) == int(metadata["feature_count"]), "scaler_mean length mismatch")
    require(len(metadata["scaler_std"]) == int(metadata["feature_count"]), "scaler_std length mismatch")
    return selected_k


def load_and_validate_models(
    output: Path,
    metadata: dict[str, Any],
) -> tuple[StandardScalerModel, PCAModel]:
    scaler = StandardScalerModel.load(spark_path(output / "scaler_model"))
    pca = PCAModel.load(spark_path(output / "pca_model"))
    comparisons = (
        (list(scaler.mean), metadata["scaler_mean"], "scaler mean"),
        (list(scaler.std), metadata["scaler_std"], "scaler std"),
        (list(pca.explainedVariance), metadata["explained_variance"], "PCA variance"),
    )
    for actual, expected, name in comparisons:
        require(len(actual) == len(expected), f"{name} length mismatch")
        difference = max(
            (abs(float(left) - float(right)) for left, right in zip(actual, expected)),
            default=0.0,
        )
        require(difference <= 1e-12, f"{name} does not match metadata")
    require(scaler.getWithMean() and scaler.getWithStd(), "wrong scaler configuration")
    require(scaler.getInputCol() == "features", "wrong scaler input column")
    require(scaler.getOutputCol() == "scaled_features", "wrong scaler output column")
    require(pca.getK() == 128, "wrong PCA component count")
    require(pca.getInputCol() == "scaled_features", "wrong PCA input column")
    require(pca.getOutputCol() == "pca_features", "wrong PCA output column")
    require(pca.pc.numRows == int(metadata["feature_count"]), "PCA row count mismatch")
    require(pca.pc.numCols == int(metadata["selected_k"]), "PCA column count mismatch")
    return scaler, pca


def validate_models(output: Path, metadata: dict[str, Any]) -> None:
    load_and_validate_models(output, metadata)


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

    null_row = F.col("track_id").isNull() | F.col(EMBEDDING_COLUMN).isNull()
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
    stats = embeddings.agg(
        F.count("*").alias("rows"),
        F.countDistinct("track_id").alias("distinct_tracks"),
        F.sum(F.when(null_row, 1).otherwise(0)).alias("null_rows"),
        F.sum(F.when(F.size(F.col(EMBEDDING_COLUMN)) != selected_k, 1).otherwise(0)).alias("bad_size"),
        F.sum(
            F.when(has_bad_value | (F.abs(norm - F.lit(1.0)) > norm_tolerance), 1).otherwise(0)
        ).alias("bad_values"),
    ).first()
    row_count = int(stats["rows"])
    distinct_tracks = int(stats["distinct_tracks"])
    null_rows = int(stats["null_rows"])
    bad_size = int(stats["bad_size"])
    bad_values = int(stats["bad_values"])

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


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def persist_frame(df: DataFrame, persisted_frames: list[DataFrame]) -> DataFrame:
    cached = df.persist(StorageLevel.MEMORY_AND_DISK)
    persisted_frames.append(cached)
    return cached


def release_frame(df: DataFrame, persisted_frames: list[DataFrame]) -> None:
    try:
        df.unpersist(blocking=False)
    except Exception as error:
        warnings.warn(f"failed to unpersist L1-1 Spark frame: {error}", RuntimeWarning)
    for index, cached in enumerate(persisted_frames):
        if cached is df:
            del persisted_frames[index]
            break


def require_columns(df: DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    require(not missing, f"{name} is missing columns: {missing}")


def array_dot(left: F.Column, right: F.Column) -> F.Column:
    products = F.zip_with(left, right, lambda x, y: x.cast("double") * y.cast("double"))
    return F.aggregate(products, F.lit(0.0), lambda total, value: total + value)


def cosine(left: F.Column, right: F.Column) -> F.Column:
    denominator = F.sqrt(array_dot(left, left) * array_dot(right, right))
    return array_dot(left, right) / denominator


def add_coverage(raw: DataFrame, feature_columns: Sequence[str]) -> DataFrame:
    availability = add_scalar_availability(raw)
    coverage_columns = tuple(
        column for column in feature_columns
        if column.startswith("has_") and column in availability.columns
    )
    require(bool(coverage_columns), "C1 feature coverage columns are unavailable")
    score = F.lit(0)
    for column in coverage_columns:
        score = score + F.when(F.col(column).cast("double") == 1.0, F.lit(1)).otherwise(F.lit(0))
    return availability.select(TRACK_ID_COLUMN, score.cast("int").alias("feature_coverage"))


def _partner_columns(df: DataFrame, window: Window) -> DataFrame:
    result = df
    for column in METADATA_COLUMNS:
        result = result.withColumn(f"candidate_{column}", F.lead(F.col(column)).over(window))
    result = result.withColumn(
        "candidate_feature_coverage", F.lead(F.col("feature_coverage")).over(window)
    )
    return result.withColumn("candidate_year_key", F.lead(F.col("year_key")).over(window))


def select_pairs(
    base: DataFrame,
    pair_type: str,
    partition_columns: Sequence[str],
    pair_count: int,
    seed: int,
) -> DataFrame:
    ordering = F.xxhash64(F.col(TRACK_ID_COLUMN), F.lit(seed))
    window = Window.partitionBy(*partition_columns).orderBy(ordering, F.col(TRACK_ID_COLUMN))
    paired = _partner_columns(base, window)
    different_song = F.col("candidate_song_id").isNotNull() & (
        F.col("song_id") != F.col("candidate_song_id")
    )
    paired = paired.where(F.col(f"candidate_{TRACK_ID_COLUMN}").isNotNull() & different_song)
    return (
        paired.select(
            F.lit(pair_type).alias("pair_type"),
            F.col(TRACK_ID_COLUMN).alias("query_track_id"),
            F.col(f"candidate_{TRACK_ID_COLUMN}").alias("candidate_track_id"),
            (F.col("year_key") == F.col("candidate_year_key")).alias("year_matched"),
            (F.col("feature_coverage") == F.col("candidate_feature_coverage")).alias(
                "coverage_matched"
            ),
            (F.col("artist_id") == F.col("candidate_artist_id")).alias("artist_matched"),
            (F.col("release_7digitalid") == F.col("candidate_release_7digitalid")).alias(
                "release_matched"
            ),
        )
        .orderBy(F.xxhash64("query_track_id", "candidate_track_id", F.lit(seed)))
        .limit(pair_count)
    )


def main() -> None:
    args = parse_args()
    validate_layout(args.output)
    metadata = read_metadata(args.output / "audio_encoder_metadata.json")
    selected_k = validate_metadata(metadata)
    require(int(metadata["row_count"]) == args.expected_rows, "metadata row_count mismatch")
    validate_c1_manifest(args.output, metadata)

    spark = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        validate_models(args.output, metadata)
        embeddings = spark.read.parquet(spark_path(args.output / "song_embeddings_audio.parquet"))
        validate_embeddings(embeddings, args.expected_rows, selected_k, args.norm_tolerance)
        print("MERLIN audio PCA validation passed.")
    finally:
        try:
            spark.stop()
        except Exception as error:
            warnings.warn(f"failed to stop Spark cleanly: {error}", RuntimeWarning)


if __name__ == "__main__":
    main()
