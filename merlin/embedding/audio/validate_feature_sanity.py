from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pyspark import StorageLevel
from pyspark.ml.feature import PCAModel, StandardScalerModel, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from artifacts import sha256_path
from columns import PREPARED_AUDIO_COLUMNS, TRACK_ID_COLUMN
from l1_stats import (
    bootstrap_hedges_g_ci,
    classify_validation,
    distribution,
    hedges_g,
    preservation_summary,
)
from preprocess import add_scalar_availability, apply_frozen_preprocess
from train_pca import (
    EMBEDDING_COLUMN,
    FEATURES_COLUMN,
    PCA_FEATURES_COLUMN,
    SCALED_FEATURES_COLUMN,
    add_normalized_embedding,
)
from validate import read_metadata, require, validate_layout, validate_metadata


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


def load_inputs(
    spark: SparkSession,
    args: argparse.Namespace,
    encoder_metadata: dict[str, Any],
    feature_columns: Sequence[str],
    persisted_frames: list[DataFrame],
) -> tuple[DataFrame, DataFrame, DataFrame, int, int]:
    raw = spark.read.parquet(spark_path(args.raw_input))
    require(tuple(raw.columns) == PREPARED_AUDIO_COLUMNS, "raw input contract mismatch")
    songs_metadata = spark.read.parquet(spark_path(args.songs_metadata))
    require_columns(songs_metadata, METADATA_COLUMNS, "songs metadata")
    saved_embeddings = spark.read.parquet(
        spark_path(args.output / "song_embeddings_audio.parquet")
    ).select(TRACK_ID_COLUMN, EMBEDDING_COLUMN)
    output_ids = persist_frame(
        saved_embeddings.select(TRACK_ID_COLUMN),
        persisted_frames,
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
    return raw, saved_embeddings, base, output_rows, base_rows


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
    args: argparse.Namespace,
    raw: DataFrame,
    saved_embeddings: DataFrame,
    pairs: DataFrame,
    encoder_metadata: dict[str, Any],
    feature_columns: Sequence[str],
    selected_k: int,
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
    scaler_model = StandardScalerModel.load(spark_path(args.output / "scaler_model"))
    pca_model = PCAModel.load(spark_path(args.output / "pca_model"))
    verify_model_metadata(encoder_metadata, scaler_model, pca_model)
    scaled = scaler_model.transform(assembled).select(TRACK_ID_COLUMN, SCALED_FEATURES_COLUMN)
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
    ).select(
        TRACK_ID_COLUMN, F.col(EMBEDDING_COLUMN).alias("final_embedding")
    )
    compared = recomputed.join(
        selected_saved,
        TRACK_ID_COLUMN,
        "inner",
    ).withColumn(
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
    compared = persist_frame(compared, persisted_frames)
    return compared, selected_count, selected_ids


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


def collect_scores(pairs: DataFrame, vectors: DataFrame, total_pairs: int) -> tuple[list[Any], int]:
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
