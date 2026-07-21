from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pyspark.ml.feature import PCAModel, StandardScalerModel, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from columns import PREPARED_AUDIO_COLUMNS, TRACK_ID_COLUMN
from frozen_preprocess import apply_frozen_preprocess
from l1_stats import bootstrap_hedges_g_ci, distribution, hedges_g, preservation_summary
from lineage import sha256_path
from preprocess import add_scalar_availability
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MERLIN C1 L1-1 feature sanity validation.")
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
        SparkSession.builder.appName("MerlinValidateC1FeatureSanity")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


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
        "candidate_feature_coverage",
        F.lead(F.col("feature_coverage")).over(window),
    )
    result = result.withColumn(
        "candidate_year_key",
        F.lead(F.col("year_key")).over(window),
    )
    return result


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
        "same_artist": select_pairs(
            same_artist, "same_artist", ("artist_id",), pair_count, seed
        ),
        "same_release": select_pairs(
            same_release, "same_release", ("release_7digitalid",), pair_count, seed + 1
        ),
        "random": select_pairs(
            base,
            "random",
            ("year_key", "feature_coverage"),
            pair_count,
            seed + 2,
        ),
    }


def verify_model_metadata(
    encoder_metadata: dict[str, Any],
    scaler_model: StandardScalerModel,
    pca_model: PCAModel,
) -> None:
    comparisons = (
        (list(scaler_model.mean), encoder_metadata["scaler_mean"], "scaler mean"),
        (list(scaler_model.std), encoder_metadata["scaler_std"], "scaler std"),
        (
            list(pca_model.explainedVariance),
            encoder_metadata["explained_variance"],
            "PCA explained variance",
        ),
    )
    for actual, expected, name in comparisons:
        require(len(actual) == len(expected), f"{name} length mismatch")
        maximum = max((abs(float(a) - float(b)) for a, b in zip(actual, expected)), default=0.0)
        require(maximum <= 1e-12, f"{name} does not match encoder metadata")
    require(pca_model.pc.numRows == int(encoder_metadata["feature_count"]), "PCA row mismatch")
    require(pca_model.pc.numCols == int(encoder_metadata["selected_k"]), "PCA column mismatch")


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
    return distributions, effects, {"pairwise_preservation": preservation, "matching_rates": matching_rates}

