from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import warnings
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pyspark import StorageLevel
from pyspark.ml.feature import PCAModel, StandardScalerModel, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType

from merlin.embedding.audio.artifacts import (
    C1_MANIFEST_NAME,
    ENCODER_METADATA_NAME,
    sha256_path,
    validate_c1_manifest,
    validate_encoder_contract,
)
from merlin.embedding.audio.columns import (
    CONTRACT_VERSION,
    PREPARED_AUDIO_COLUMNS,
    TRACK_ID_COLUMN,
    TIME_SIGNATURE_UNKNOWN_COLUMN,
    TIME_SIGNATURE_VALUES,
    build_feature_columns,
    time_signature_one_hot_column,
)
from merlin.embedding.audio.l1_stats import (
    bootstrap_hedges_g_ci,
    classify_validation,
    distribution,
    hedges_g,
    preservation_summary,
)
from merlin.embedding.audio.preprocess import (
    SEGMENT_MEDIAN_BATCH_SIZE,
    add_scalar_availability,
    apply_frozen_preprocess,
)
from merlin.embedding.audio.train_pca import (
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
    parser.add_argument(
        "--allow-noncanonical-dimension",
        action="store_true",
        help="Allow an isolated artifact-only smoke run whose PCA dimension is not 128.",
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
        ENCODER_METADATA_NAME,
        "pca_model",
        C1_MANIFEST_NAME,
        "scaler_model",
    )
    missing = [name for name in required if not (output_dir / name).exists()]
    require(not missing, f"audio output missing files: {missing}")


def validate_metadata(
    metadata: dict[str, Any],
    *,
    require_canonical_dimension: bool = True,
) -> int:
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
        "target_variance",
        "fixed_k",
        "max_components",
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
    encoder = validate_encoder_contract(
        metadata,
        require_canonical_dimension=require_canonical_dimension,
    )
    selected_k = encoder.selected_k
    require(metadata["merlin_schema_version"] == "3.0", "wrong MERLIN schema version")
    require(metadata["shared_audio_contract_version"] == CONTRACT_VERSION, "wrong audio contract")
    require(int(metadata["c1_feature_version"]) == 2, "wrong C1 feature version")
    require(metadata["model_ready_schema_version"] == "c1_model_ready_v2", "wrong model schema")
    require(metadata["permanent_dropped_fields"] == ["danceability", "energy"], "wrong dropped fields")
    require(int(metadata["shared_audio_feature_count"]) == 628, "shared feature count mismatch")
    require(int(metadata["merlin_array_feature_count"]) == 552, "array feature count mismatch")
    require(int(metadata["merlin_raw_view_count"]) == 563, "raw view count mismatch")
    require(metadata["embedding_format"] == "array<float32>", "wrong embedding format")
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


def build_pairs(base: DataFrame, pair_count: int, seed: int) -> dict[str, DataFrame]:
    same_artist = base.where(F.col("artist_id").isNotNull() & (F.length("artist_id") > 0))
    same_release = base.where(
        F.col("release_7digitalid").isNotNull() & (F.col("release_7digitalid") > 0)
    )
    return {
        "same_artist": select_pairs(same_artist, "same_artist", ("artist_id",), pair_count, seed),
        "same_release": select_pairs(
            same_release, "same_release", ("release_7digitalid",), pair_count, seed + 1
        ),
        "random": select_pairs(
            base, "random", ("year_key", "feature_coverage"), pair_count, seed + 2
        ),
    }


def score_pairs(pairs: DataFrame, vectors: DataFrame) -> DataFrame:
    query = vectors.select(
        F.col(TRACK_ID_COLUMN).alias("query_track_id"),
        F.col("pre_pca_vector").alias("query_pre"),
        F.col("final_embedding").alias("query_post"),
    )
    candidate = vectors.select(
        F.col(TRACK_ID_COLUMN).alias("candidate_track_id"),
        F.col("pre_pca_vector").alias("candidate_pre"),
        F.col("final_embedding").alias("candidate_post"),
    )
    joined = pairs.join(query, "query_track_id", "inner").join(
        candidate, "candidate_track_id", "inner"
    )
    return joined.select(
        "pair_type",
        "query_track_id",
        "candidate_track_id",
        "year_matched",
        "coverage_matched",
        "artist_matched",
        "release_matched",
        cosine(F.col("query_pre"), F.col("candidate_pre")).alias("pre_pca_cosine"),
        cosine(F.col("query_post"), F.col("candidate_post")).alias("pca_128_cosine"),
    )


def summarize_scores(
    rows: list[Any],
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    samples: dict[str, dict[str, list[float]]] = {
        representation: {pair_type: [] for pair_type in PAIR_TYPES}
        for representation in ("pre_pca", "pca_128")
    }
    matching: dict[str, dict[str, int]] = {
        pair_type: {
            "year_matched": 0,
            "coverage_matched": 0,
            "artist_matched": 0,
            "release_matched": 0,
        }
        for pair_type in PAIR_TYPES
    }
    for row in rows:
        pair_type = row["pair_type"]
        samples["pre_pca"][pair_type].append(float(row["pre_pca_cosine"]))
        samples["pca_128"][pair_type].append(float(row["pca_128_cosine"]))
        for name in matching[pair_type]:
            matching[pair_type][name] += int(bool(row[name]))

    distributions = {
        representation: {
            pair_type: distribution(samples[representation][pair_type])
            for pair_type in PAIR_TYPES
        }
        for representation in samples
    }
    effects: dict[str, Any] = {}
    for representation_index, representation in enumerate(("pre_pca", "pca_128")):
        effects[representation] = {}
        random_sample = samples[representation]["random"]
        for relation_index, relation in enumerate(("same_artist", "same_release")):
            value = hedges_g(samples[representation][relation], random_sample)
            interval = bootstrap_hedges_g_ci(
                samples[representation][relation],
                random_sample,
                bootstrap_samples,
                seed + representation_index * 100 + relation_index,
            )
            effects[representation][relation] = {
                "hedges_g": value,
                "bootstrap_95_ci": interval,
            }
    preservation = {
        pair_type: preservation_summary(
            samples["pre_pca"][pair_type], samples["pca_128"][pair_type]
        )
        for pair_type in PAIR_TYPES
    }
    matching_rates = {
        pair_type: {
            name: count / len(samples["pre_pca"][pair_type])
            for name, count in values.items()
        }
        for pair_type, values in matching.items()
    }
    return distributions, effects, {
        "pairwise_preservation": preservation,
        "matching_rates": matching_rates,
    }


def conclusion(effects: dict[str, Any], formal: bool) -> dict[str, Any]:
    criterion = (
        "For same_artist and same_release, the pre-PCA and PCA-128 Hedges' g "
        "bootstrap 95% CI lower bounds must all be greater than zero."
    )
    if not formal:
        return {
            "eligible": False,
            "supported": None,
            "criterion": criterion,
            "statement": "Smoke data validates execution logic only; it is not formal L1-1 evidence.",
        }
    intervals = [
        effects[representation][relation]["bootstrap_95_ci"]
        for representation in ("pre_pca", "pca_128")
        for relation in ("same_artist", "same_release")
    ]
    supported = all(interval is not None and interval[0] > 0.0 for interval in intervals)
    statement = (
        "C1 preserves metadata-correlated acoustic structure after deterministic "
        "preprocessing and PCA."
        if supported
        else "The formal L1-1 measurements do not support the permitted C1 conclusion."
    )
    return {
        "eligible": True,
        "supported": supported,
        "criterion": criterion,
        "statement": statement,
    }


def load_l1_inputs(
    spark: SparkSession,
    args: argparse.Namespace,
    encoder_metadata: dict[str, Any],
    feature_columns: Sequence[str],
    saved_embeddings: DataFrame,
    persisted_frames: list[DataFrame],
) -> tuple[DataFrame, DataFrame, int, int]:
    raw = spark.read.parquet(spark_path(args.raw_input))
    require(tuple(raw.columns) == PREPARED_AUDIO_COLUMNS, "raw input contract mismatch")
    songs_metadata = spark.read.parquet(spark_path(args.songs_metadata))
    require_columns(songs_metadata, METADATA_COLUMNS, "songs metadata")
    output_ids = persist_frame(
        saved_embeddings.select(TRACK_ID_COLUMN), persisted_frames
    )
    output_rows = output_ids.count()
    require(output_rows == int(encoder_metadata["row_count"]), "embedding row count mismatch")

    id_lookup = F.broadcast(output_ids) if output_rows <= 100_000 else output_ids
    coverage = add_coverage(raw.join(id_lookup, TRACK_ID_COLUMN, "inner"), feature_columns)
    base = persist_frame(
        songs_metadata.select(*METADATA_COLUMNS)
        .join(id_lookup, TRACK_ID_COLUMN, "inner")
        .join(coverage, TRACK_ID_COLUMN, "inner")
        .withColumn(
            "year_key",
            F.when(
                (F.col("has_year") == 1) & F.col("year").isNotNull(), F.col("year")
            ).otherwise(F.lit(0)),
        ),
        persisted_frames,
    )
    base_rows = base.count()
    require(base_rows == output_rows, "songs metadata/raw coverage does not cover C1 output")
    release_frame(output_ids, persisted_frames)
    return raw, base, output_rows, base_rows


def prepare_pairs(
    base: DataFrame,
    args: argparse.Namespace,
    persisted_frames: list[DataFrame],
) -> tuple[DataFrame, dict[str, int], int]:
    pair_frames = {
        name: persist_frame(frame, persisted_frames)
        for name, frame in build_pairs(base, args.pair_count, args.seed).items()
    }
    pair_counts = {name: frame.count() for name, frame in pair_frames.items()}
    for name, count in pair_counts.items():
        require(count > 0, f"no eligible {name} pairs")
        if not args.allow_partial_pairs:
            require(count == args.pair_count, f"{name} has {count}, expected {args.pair_count} pairs")
    pairs = pair_frames[PAIR_TYPES[0]]
    for pair_type in PAIR_TYPES[1:]:
        pairs = pairs.unionByName(pair_frames[pair_type])
    pairs = persist_frame(pairs, persisted_frames)
    total_pairs = pairs.count()
    require(total_pairs == sum(pair_counts.values()), "pair union count mismatch")
    for frame in pair_frames.values():
        release_frame(frame, persisted_frames)
    release_frame(base, persisted_frames)
    return pairs, pair_counts, total_pairs


def recompute_embeddings(
    raw: DataFrame,
    saved_embeddings: DataFrame,
    pairs: DataFrame,
    encoder_metadata: dict[str, Any],
    feature_columns: Sequence[str],
    selected_k: int,
    scaler_model: StandardScalerModel,
    pca_model: PCAModel,
    persisted_frames: list[DataFrame],
) -> tuple[DataFrame, int, DataFrame]:
    selected_ids = persist_frame(
        pairs.select(F.col("query_track_id").alias(TRACK_ID_COLUMN)).union(
            pairs.select(F.col("candidate_track_id").alias(TRACK_ID_COLUMN))
        ).distinct(),
        persisted_frames,
    )
    selected_count = selected_ids.count()
    selected_raw = raw.join(F.broadcast(selected_ids), TRACK_ID_COLUMN, "inner")
    processed = apply_frozen_preprocess(
        selected_raw, feature_columns, encoder_metadata["preprocess"]
    )
    assembled = VectorAssembler(
        inputCols=list(feature_columns), outputCol=FEATURES_COLUMN
    ).transform(processed).select(TRACK_ID_COLUMN, FEATURES_COLUMN)
    scaled = scaler_model.transform(assembled).select(
        TRACK_ID_COLUMN, SCALED_FEATURES_COLUMN
    )
    projected = pca_model.transform(scaled).select(
        TRACK_ID_COLUMN, SCALED_FEATURES_COLUMN, PCA_FEATURES_COLUMN
    )
    recomputed = add_normalized_embedding(projected, selected_k).select(
        TRACK_ID_COLUMN,
        vector_to_array(F.col(SCALED_FEATURES_COLUMN)).alias("pre_pca_vector"),
        F.col(EMBEDDING_COLUMN).alias("recomputed_embedding"),
    )
    selected_saved = saved_embeddings.join(
        F.broadcast(selected_ids), TRACK_ID_COLUMN, "inner"
    ).select(TRACK_ID_COLUMN, F.col(EMBEDDING_COLUMN).alias("final_embedding"))
    compared = recomputed.join(selected_saved, TRACK_ID_COLUMN, "inner").withColumn(
        "maximum_difference",
        F.array_max(
            F.zip_with(
                "recomputed_embedding",
                "final_embedding",
                lambda left, right: F.abs(left.cast("double") - right.cast("double")),
            )
        ),
    ).select(
        TRACK_ID_COLUMN,
        "pre_pca_vector",
        "final_embedding",
        "maximum_difference",
    )
    return persist_frame(compared, persisted_frames), selected_count, selected_ids


def validate_reproduction(
    compared: DataFrame,
    selected_count: int,
    tolerance: float,
) -> tuple[DataFrame, float]:
    stats = compared.agg(
        F.count("*").alias("rows"),
        F.max("maximum_difference").alias("maximum_difference"),
    ).first()
    require(int(stats["rows"]) == selected_count, "selected vectors are incomplete")
    maximum_difference = float(stats["maximum_difference"])
    require(
        math.isfinite(maximum_difference) and maximum_difference <= tolerance,
        "frozen preprocessing/PCA does not reproduce saved embeddings",
    )
    vectors = compared.select(TRACK_ID_COLUMN, "pre_pca_vector", "final_embedding")
    return vectors, maximum_difference


def collect_scores(
    pairs: DataFrame,
    vectors: DataFrame,
    total_pairs: int,
) -> tuple[list[Any], int]:
    rows = score_pairs(pairs, vectors).collect()
    scored_count = len(rows)
    require(scored_count == total_pairs, "not every selected pair was scored")
    require(
        all(
            row[name] is not None and math.isfinite(float(row[name]))
            for row in rows
            for name in ("pre_pca_cosine", "pca_128_cosine")
        ),
        "L1-1 produced a non-finite cosine",
    )
    return rows, scored_count


def build_report(
    args: argparse.Namespace,
    encoder_metadata_path: Path,
    encoder_metadata: dict[str, Any],
    pair_counts: dict[str, int],
    distributions: dict[str, Any],
    effects: dict[str, Any],
    diagnostics: dict[str, Any],
    output_rows: int,
    base_rows: int,
    selected_count: int,
    scored_count: int,
    maximum_difference: float,
) -> dict[str, Any]:
    pair_requirement_met = all(
        pair_counts.get(pair_type) == args.pair_count for pair_type in PAIR_TYPES
    )
    finding = conclusion(effects, pair_requirement_met and not args.allow_partial_pairs)
    formal, status = classify_validation(
        pair_counts,
        PAIR_TYPES,
        args.pair_count,
        args.allow_partial_pairs,
        finding["supported"] is True,
    )
    return {
        "artifact_type": "c1_l1_feature_sanity_report",
        "validation_version": VALIDATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": status,
        "formal_pair_requirement_met": pair_requirement_met,
        "parameters": {
            "target_pairs_per_type": args.pair_count,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_method": "independent percentile bootstrap of Hedges' g",
            "confidence_level": 0.95,
            "seed": args.seed,
            "allow_partial_pairs": args.allow_partial_pairs,
            "random_matching": "exact year_key and feature-availability count",
        },
        "inputs": {
            "raw_input": str(args.raw_input.resolve()),
            "songs_metadata": str(args.songs_metadata.resolve()),
            "audio_output": str(args.output.resolve()),
            "audio_encoder_metadata_sha256": sha256_path(encoder_metadata_path),
            "shared_audio_contract_version": encoder_metadata["shared_audio_contract_version"],
            "c1_feature_version": encoder_metadata["c1_feature_version"],
        },
        "integrity": {
            "embedding_rows": output_rows,
            "eligible_rows": base_rows,
            "selected_tracks": selected_count,
            "scored_pairs": scored_count,
            "frozen_transform_max_abs_embedding_error": maximum_difference,
            "frozen_transform_tolerance": args.reproduction_tolerance,
        },
        "pair_counts": pair_counts,
        "distributions": distributions,
        "effect_size_vs_random": {
            "definition": "Hedges' g standardized mean difference; positive favors the relation",
            **effects,
        },
        **diagnostics,
        "conclusion": finding,
    }


def validate_c1_artifacts(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    selected_k: int,
    embeddings: DataFrame,
) -> None:
    require(int(metadata["row_count"]) == args.expected_rows, "metadata row_count mismatch")
    validate_c1_manifest(args.output, metadata)
    validate_embeddings(embeddings, args.expected_rows, selected_k, args.norm_tolerance)
    print("MERLIN audio PCA validation passed.")


def validate_l1_feature_sanity(
    spark: SparkSession,
    args: argparse.Namespace,
    encoder_metadata_path: Path,
    encoder_metadata: dict[str, Any],
    selected_k: int,
    scaler_model: StandardScalerModel,
    pca_model: PCAModel,
    saved_embeddings: DataFrame,
) -> None:
    feature_columns = tuple(encoder_metadata["feature_columns"])
    persisted_frames: list[DataFrame] = []
    try:
        raw, base, output_rows, base_rows = load_l1_inputs(
            spark,
            args,
            encoder_metadata,
            feature_columns,
            saved_embeddings,
            persisted_frames,
        )
        pairs, pair_counts, total_pairs = prepare_pairs(base, args, persisted_frames)
        compared, selected_count, selected_ids = recompute_embeddings(
            raw,
            saved_embeddings,
            pairs,
            encoder_metadata,
            feature_columns,
            selected_k,
            scaler_model,
            pca_model,
            persisted_frames,
        )
        vectors, maximum_difference = validate_reproduction(
            compared, selected_count, args.reproduction_tolerance
        )
        release_frame(selected_ids, persisted_frames)
        score_rows, scored_count = collect_scores(pairs, vectors, total_pairs)
        release_frame(pairs, persisted_frames)
        release_frame(compared, persisted_frames)
        distributions, effects, diagnostics = summarize_scores(
            score_rows, args.bootstrap_samples, args.seed
        )
        report = build_report(
            args,
            encoder_metadata_path,
            encoder_metadata,
            pair_counts,
            distributions,
            effects,
            diagnostics,
            output_rows,
            base_rows,
            selected_count,
            scored_count,
            maximum_difference,
        )
        report_path = args.report or args.output / "validation_report.json"
        write_json(report, report_path)
        require(
            report["validation_status"] != "FAIL",
            f"{report['conclusion']['statement']} Report: {report_path}",
        )
        print(
            "c1_l1_feature_sanity_passed "
            f"status={report['validation_status']}, pairs={pair_counts}, "
            f"selected_tracks={selected_count}, reproduction_error={maximum_difference:.3g}, "
            f"report={report_path}"
        )
    finally:
        for frame in reversed(persisted_frames.copy()):
            release_frame(frame, persisted_frames)


def main() -> None:
    args = parse_args()
    require(args.expected_rows > 0, "expected rows must be positive")
    require(args.shuffle_partitions > 0, "shuffle partitions must be positive")
    require(args.norm_tolerance > 0.0, "norm tolerance must be positive")
    if args.mode in {"l1", "all"}:
        require(args.pair_count > 0, "pair_count must be positive")
        require(args.bootstrap_samples > 0, "bootstrap_samples must be positive")
        require(args.reproduction_tolerance > 0.0, "reproduction_tolerance must be positive")
    validate_layout(args.output)
    encoder_metadata_path = args.output / ENCODER_METADATA_NAME
    metadata = read_metadata(encoder_metadata_path)
    require_canonical_dimension = (
        args.mode in {"l1", "all"} or not args.allow_noncanonical_dimension
    )
    selected_k = validate_metadata(
        metadata,
        require_canonical_dimension=require_canonical_dimension,
    )

    spark = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        scaler_model, pca_model = load_and_validate_models(args.output, metadata)
        embeddings = spark.read.parquet(spark_path(args.output / "song_embeddings_audio.parquet"))
        if args.mode in {"artifact", "all"}:
            validate_c1_artifacts(args, metadata, selected_k, embeddings)
        if args.mode in {"l1", "all"}:
            validate_l1_feature_sanity(
                spark,
                args,
                encoder_metadata_path,
                metadata,
                selected_k,
                scaler_model,
                pca_model,
                embeddings.select(TRACK_ID_COLUMN, EMBEDDING_COLUMN),
            )
    finally:
        try:
            spark.stop()
        except Exception as error:
            warnings.warn(f"failed to stop Spark cleanly: {error}", RuntimeWarning)


if __name__ == "__main__":
    main()
