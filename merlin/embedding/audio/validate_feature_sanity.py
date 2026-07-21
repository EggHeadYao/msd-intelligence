from __future__ import annotations

import argparse
import json
import math
import warnings
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
) -> tuple[DataFrame, DataFrame, DataFrame, int, int]:
    raw = spark.read.parquet(spark_path(args.raw_input))
    require(tuple(raw.columns) == PREPARED_AUDIO_COLUMNS, "raw input contract mismatch")
    songs_metadata = spark.read.parquet(spark_path(args.songs_metadata))
    require_columns(songs_metadata, METADATA_COLUMNS, "songs metadata")
    saved_embeddings = spark.read.parquet(
        spark_path(args.output / "song_embeddings_audio.parquet")
    ).select(TRACK_ID_COLUMN, EMBEDDING_COLUMN)
    output_rows = saved_embeddings.count()
    require(output_rows == int(encoder_metadata["row_count"]), "embedding row count mismatch")

    output_ids = saved_embeddings.select(TRACK_ID_COLUMN)
    id_lookup = F.broadcast(output_ids) if output_rows <= 100_000 else output_ids
    coverage = add_coverage(raw.join(id_lookup, TRACK_ID_COLUMN, "inner"), feature_columns)
    base = (
        songs_metadata.select(*METADATA_COLUMNS)
        .join(id_lookup, TRACK_ID_COLUMN, "inner")
        .join(coverage, TRACK_ID_COLUMN, "inner")
        .withColumn(
            "year_key",
            F.when(
                (F.col("has_year") == 1) & F.col("year").isNotNull(), F.col("year")
            ).otherwise(F.lit(0)),
        )
    ).cache()
    base_rows = base.count()
    require(base_rows == output_rows, "songs metadata/raw coverage does not cover C1 output")
    return raw, saved_embeddings, base, output_rows, base_rows


def prepare_pairs(base: DataFrame, args: argparse.Namespace) -> tuple[DataFrame, dict[str, int], int]:
    pair_frames = {
        name: frame.cache()
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
    pairs = pairs.cache()
    total_pairs = pairs.count()
    require(total_pairs == sum(pair_counts.values()), "pair union count mismatch")
    return pairs, pair_counts, total_pairs


def recompute_embeddings(
    args: argparse.Namespace,
    raw: DataFrame,
    saved_embeddings: DataFrame,
    pairs: DataFrame,
    encoder_metadata: dict[str, Any],
    feature_columns: Sequence[str],
    selected_k: int,
) -> tuple[DataFrame, int]:
    selected_ids = pairs.select(F.col("query_track_id").alias(TRACK_ID_COLUMN)).union(
        pairs.select(F.col("candidate_track_id").alias(TRACK_ID_COLUMN))
    ).distinct()
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
    compared = recomputed.join(
        saved_embeddings.select(
            TRACK_ID_COLUMN, F.col(EMBEDDING_COLUMN).alias("final_embedding")
        ),
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
    )
    return compared, selected_ids.count()


def validate_reproduction(
    compared: DataFrame,
    selected_count: int,
    tolerance: float,
) -> tuple[DataFrame, float]:
    require(compared.count() == selected_count, "selected vectors are incomplete")
    maximum_difference = float(
        compared.agg(F.max("maximum_difference").alias("value")).first()["value"]
    )
    require(
        math.isfinite(maximum_difference) and maximum_difference <= tolerance,
        "frozen preprocessing/PCA does not reproduce saved embeddings",
    )
    vectors = compared.select(TRACK_ID_COLUMN, "pre_pca_vector", "final_embedding").cache()
    return vectors, maximum_difference


def collect_scores(pairs: DataFrame, vectors: DataFrame, total_pairs: int) -> tuple[list[Any], int]:
    scored = score_pairs(pairs, vectors).cache()
    scored_count = scored.count()
    require(scored_count == total_pairs, "not every selected pair was scored")
    invalid_scores = scored.where(
        F.col("pre_pca_cosine").isNull()
        | F.isnan("pre_pca_cosine")
        | (F.abs("pre_pca_cosine") == float("inf"))
        | F.col("pca_128_cosine").isNull()
        | F.isnan("pca_128_cosine")
        | (F.abs("pca_128_cosine") == float("inf"))
    ).limit(1).count()
    require(invalid_scores == 0, "L1-1 produced a non-finite cosine")
    return scored.collect(), scored_count


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
    formal = all(count == args.pair_count for count in pair_counts.values())
    return {
        "artifact_type": "c1_l1_feature_sanity_report",
        "validation_version": VALIDATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": "PASS" if formal else "SMOKE_PASS",
        "formal_pair_requirement_met": formal,
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
        "conclusion": conclusion(effects, formal),
    }


def main() -> None:
    args = parse_args()
    require(args.pair_count > 0, "pair_count must be positive")
    require(args.bootstrap_samples > 0, "bootstrap_samples must be positive")
    require(args.reproduction_tolerance > 0.0, "reproduction_tolerance must be positive")
    validate_layout(args.output)
    encoder_metadata_path = args.output / "audio_encoder_metadata.json"
    encoder_metadata = read_metadata(encoder_metadata_path)
    selected_k = validate_metadata(encoder_metadata)
    feature_columns = tuple(encoder_metadata["feature_columns"])

    spark = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw, saved, base, output_rows, base_rows = load_inputs(
            spark, args, encoder_metadata, feature_columns
        )
        pairs, pair_counts, total_pairs = prepare_pairs(base, args)
        compared, selected_count = recompute_embeddings(
            args, raw, saved, pairs, encoder_metadata, feature_columns, selected_k
        )
        vectors, maximum_difference = validate_reproduction(
            compared, selected_count, args.reproduction_tolerance
        )
        score_rows, scored_count = collect_scores(pairs, vectors, total_pairs)
        distributions, effects, diagnostics = summarize_scores(
            score_rows, args.bootstrap_samples, args.seed
        )
        report = build_report(
            args, encoder_metadata_path, encoder_metadata, pair_counts,
            distributions, effects, diagnostics, output_rows, base_rows,
            selected_count, scored_count, maximum_difference,
        )
        report_path = args.report or args.output / "validation_report.json"
        write_json(report, report_path)
        print(
            "c1_l1_feature_sanity_passed "
            f"status={report['validation_status']}, pairs={pair_counts}, "
            f"selected_tracks={selected_count}, reproduction_error={maximum_difference:.3g}, "
            f"report={report_path}"
        )
    finally:
        try:
            spark.stop()
        except Exception as error:
            warnings.warn(f"failed to stop Spark cleanly: {error}", RuntimeWarning)


if __name__ == "__main__":
    main()
