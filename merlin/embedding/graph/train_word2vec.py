"""Train the canonical MERLIN C2 Word2Vec track embeddings."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType

from merlin.embedding.graph.config import (
    NUM_WALKS,
    SEED,
    WORD2VEC_MAX_ITER,
    WORD2VEC_MAX_SENTENCE_LENGTH,
    WORD2VEC_MIN_COUNT,
    WORD2VEC_NUM_PARTITIONS,
    WORD2VEC_STEP_SIZE,
    WORD2VEC_VECTOR_SIZE,
    WORD2VEC_WINDOW_SIZE,
)


EMBEDDINGS_NAME = "song_embeddings_graph.parquet"
MODEL_NAME = "word2vec_model"
METADATA_NAME = "graph_encoder_metadata.json"
REQUIRED_WALK_COLUMNS = {
    "track_id",
    "walk_id",
    "walk_seq",
    "walk_len",
    "transition_count",
    "path_counts",
    "path_eligible_counts",
    "termination_reason",
}
NORM_EPSILON = 1e-12
NORM_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vector-size", type=int, default=WORD2VEC_VECTOR_SIZE)
    parser.add_argument("--window-size", type=int, default=WORD2VEC_WINDOW_SIZE)
    parser.add_argument("--min-count", type=int, default=WORD2VEC_MIN_COUNT)
    parser.add_argument("--max-iter", type=int, default=WORD2VEC_MAX_ITER)
    parser.add_argument("--step-size", type=float, default=WORD2VEC_STEP_SIZE)
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=WORD2VEC_NUM_PARTITIONS,
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=WORD2VEC_MAX_SENTENCE_LENGTH,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--expected-tracks", type=int, default=1_000_000)
    parser.add_argument("--walks-per-track", type=int, default=NUM_WALKS)
    parser.add_argument("--limit-tracks", type=int, default=0)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--output-partitions", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def create_spark(shuffle_partitions: int) -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    return (
        SparkSession.builder.appName("MerlinC2Word2Vec")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.pyspark.python", sys.executable)
        .getOrCreate()
    )


def validate_parameters(args: argparse.Namespace) -> None:
    require(args.vector_size > 0, "vector size must be positive")
    require(args.window_size > 0, "window size must be positive")
    require(args.min_count >= 0, "min count must be non-negative")
    require(args.max_iter > 0, "max iterations must be positive")
    require(args.step_size > 0.0, "step size must be positive")
    require(args.num_partitions > 0, "Word2Vec partitions must be positive")
    require(args.max_sentence_length > 0, "maximum sentence length must be positive")
    require(args.expected_tracks > 0, "expected track count must be positive")
    require(args.walks_per_track > 0, "walks per track must be positive")
    require(args.limit_tracks >= 0, "track limit must be non-negative")
    require(args.shuffle_partitions > 0, "shuffle partitions must be positive")
    require(args.output_partitions > 0, "output partitions must be positive")


def prepare_output(output: Path, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    targets = (output / EMBEDDINGS_NAME, output / MODEL_NAME, output / METADATA_NAME)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"C2 Word2Vec output already exists: {names}")
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def read_walks(
    spark: SparkSession,
    path: Path,
    limit_tracks: int,
) -> DataFrame:
    walks = spark.read.parquet(spark_path(path))
    missing = REQUIRED_WALK_COLUMNS - set(walks.columns)
    require(not missing, f"walk input is missing columns: {sorted(missing)}")

    track_field = walks.schema["track_id"]
    walk_field = walks.schema["walk_seq"]
    require(isinstance(track_field.dataType, StringType), "track_id must be string")
    require(
        isinstance(walk_field.dataType, ArrayType)
        and isinstance(walk_field.dataType.elementType, IntegerType),
        "walk_seq must be array<int>",
    )

    if limit_tracks > 0:
        selected_tracks = (
            walks.select("track_id").distinct().orderBy("track_id").limit(limit_tracks)
        )
        walks = walks.join(F.broadcast(selected_tracks), "track_id", "inner")
    return walks


def validate_walks(
    walks: DataFrame,
    expected_tracks: int,
    walks_per_track: int,
    max_sentence_length: int,
) -> dict[str, int]:
    path_count_total = F.aggregate(
        F.col("path_counts"),
        F.lit(0),
        lambda total, value: total + value,
    )
    invalid_condition = (
        F.col("track_id").isNull()
        | (F.length("track_id") == 0)
        | F.col("walk_id").isNull()
        | (F.col("walk_id") < 0)
        | (F.col("walk_id") >= walks_per_track)
        | F.col("walk_seq").isNull()
        | F.col("walk_len").isNull()
        | F.col("transition_count").isNull()
        | F.col("path_counts").isNull()
        | F.col("path_eligible_counts").isNull()
        | F.col("termination_reason").isNull()
        | (F.size("walk_seq") < 1)
        | (F.size("walk_seq") > max_sentence_length)
        | F.exists("walk_seq", lambda value: value.isNull() | (value < 0))
        | (F.col("walk_len") != F.size("walk_seq"))
        | (F.col("transition_count") != F.size("walk_seq") - 1)
        | (F.size("path_counts") != 4)
        | (F.size("path_eligible_counts") != 4)
        | F.exists("path_counts", lambda value: value.isNull() | (value < 0))
        | F.exists(
            "path_eligible_counts",
            lambda value: value.isNull() | (value < 0),
        )
        | (path_count_total != F.col("transition_count"))
        | ~F.col("termination_reason").isin("target_length", "no_eligible_path")
    )
    require(
        walks.where(invalid_condition).limit(1).count() == 0,
        "walk input violates the canonical schema or value contract",
    )

    root_node = F.element_at("walk_seq", 1)
    summary = walks.agg(
        F.count("*").alias("rows"),
        F.countDistinct("track_id").alias("tracks"),
        F.countDistinct(root_node).alias("root_nodes"),
        F.countDistinct(F.struct("track_id", "walk_id")).alias("walk_keys"),
        F.min("walk_len").alias("min_walk_len"),
        F.max("walk_len").alias("max_walk_len"),
    ).first()
    require(summary is not None, "walk input is empty")
    rows = int(summary["rows"])
    tracks = int(summary["tracks"])
    root_nodes = int(summary["root_nodes"])
    walk_keys = int(summary["walk_keys"])
    require(tracks == expected_tracks, "walk track count does not match expectation")
    require(
        root_nodes == expected_tracks, "walk root IDs are not one-to-one with tracks"
    )
    require(
        rows == expected_tracks * walks_per_track,
        "walk row count does not match tracks times walks-per-track",
    )
    require(walk_keys == rows, "track_id and walk_id pairs are not unique")

    per_track = (
        walks.groupBy("track_id")
        .count()
        .agg(
            F.min("count").alias("minimum"),
            F.max("count").alias("maximum"),
        )
        .first()
    )
    require(per_track is not None, "walk input has no track groups")
    require(
        int(per_track["minimum"]) == walks_per_track
        and int(per_track["maximum"]) == walks_per_track,
        "each track must have exactly walks-per-track rows",
    )
    return {
        "rows": rows,
        "tracks": tracks,
        "root_nodes": root_nodes,
        "walk_keys": walk_keys,
        "min_walk_len": int(summary["min_walk_len"]),
        "max_walk_len": int(summary["max_walk_len"]),
    }


def make_root_mapping(walks: DataFrame) -> DataFrame:
    mapping = walks.select(
        F.element_at("walk_seq", 1).cast("int").alias("node_id"),
        "track_id",
    ).distinct()
    collisions = mapping.groupBy("node_id").count().where(F.col("count") != 1)
    require(collisions.limit(1).count() == 0, "walk root node maps to multiple tracks")
    return mapping.persist(StorageLevel.DISK_ONLY)


def make_corpus(walks: DataFrame) -> DataFrame:
    return walks.select(
        F.transform("walk_seq", lambda node: node.cast("string")).alias("tokens"),
    )


def fit_word2vec(corpus: DataFrame, args: argparse.Namespace) -> Word2VecModel:
    estimator = Word2Vec(
        vectorSize=args.vector_size,
        windowSize=args.window_size,
        minCount=args.min_count,
        maxIter=args.max_iter,
        stepSize=args.step_size,
        numPartitions=args.num_partitions,
        maxSentenceLength=args.max_sentence_length,
        seed=args.seed,
        inputCol="tokens",
        outputCol="sentence_embedding",
    )
    return estimator.fit(corpus)


def normalize_track_vectors(
    model: Word2VecModel,
    root_mapping: DataFrame,
) -> DataFrame:
    raw_vectors = model.getVectors().select(
        F.col("word").cast("int").alias("node_id"),
        vector_to_array("vector", "float32").alias("raw_embedding"),
    )
    require(
        raw_vectors.where(F.col("node_id").isNull()).limit(1).count() == 0,
        "Word2Vec emitted a non-integer token",
    )

    joined = root_mapping.join(raw_vectors, "node_id", "left")
    require(
        joined.where(F.col("raw_embedding").isNull()).limit(1).count() == 0,
        "Word2Vec is missing one or more root-track vectors",
    )
    squared_norm = F.aggregate(
        F.col("raw_embedding"),
        F.lit(0.0),
        lambda total, value: total + value.cast("double") * value.cast("double"),
    )
    norm = F.sqrt(squared_norm)
    return joined.select(
        "node_id",
        "track_id",
        F.transform(
            "raw_embedding",
            lambda value: (value.cast("double") / norm).cast("float"),
        ).alias("embedding"),
        norm.alias("raw_norm"),
    )


def validate_embeddings(
    embeddings: DataFrame,
    expected_tracks: int,
    vector_size: int,
) -> dict[str, float | int]:
    normalized_norm = F.sqrt(
        F.aggregate(
            F.col("embedding"),
            F.lit(0.0),
            lambda total, value: total + value.cast("double") * value.cast("double"),
        ),
    )
    invalid_value = F.exists(
        "embedding",
        lambda value: (
            value.isNull()
            | F.isnan(value.cast("double"))
            | (F.abs(value.cast("double")) == F.lit(float("inf")))
        ),
    )
    invalid = embeddings.where(
        F.col("node_id").isNull()
        | F.col("track_id").isNull()
        | F.col("embedding").isNull()
        | (F.size("embedding") != vector_size)
        | (F.col("raw_norm") <= NORM_EPSILON)
        | invalid_value
        | (F.abs(normalized_norm - 1.0) > NORM_TOLERANCE),
    )
    require(invalid.limit(1).count() == 0, "graph embeddings failed numeric validation")

    summary = embeddings.agg(
        F.count("*").alias("rows"),
        F.countDistinct("node_id").alias("nodes"),
        F.countDistinct("track_id").alias("tracks"),
        F.min("raw_norm").alias("min_raw_norm"),
        F.max("raw_norm").alias("max_raw_norm"),
        F.min(normalized_norm).alias("min_norm"),
        F.max(normalized_norm).alias("max_norm"),
    ).first()
    require(summary is not None, "graph embedding output is empty")
    require(int(summary["rows"]) == expected_tracks, "embedding row count mismatch")
    require(
        int(summary["nodes"]) == expected_tracks, "embedding node IDs are not unique"
    )
    require(
        int(summary["tracks"]) == expected_tracks, "embedding track IDs are not unique"
    )
    return {
        "rows": int(summary["rows"]),
        "distinct_node_ids": int(summary["nodes"]),
        "distinct_track_ids": int(summary["tracks"]),
        "min_raw_norm": float(summary["min_raw_norm"]),
        "max_raw_norm": float(summary["max_raw_norm"]),
        "min_norm": float(summary["min_norm"]),
        "max_norm": float(summary["max_norm"]),
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    validate_parameters(args)
    effective_tracks = args.limit_tracks or args.expected_tracks
    if args.limit_tracks:
        require(
            args.limit_tracks == args.expected_tracks,
            "when limit-tracks is set, expected-tracks must equal the limit",
        )
    prepare_output(args.output, args.overwrite)

    spark = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    started = time.monotonic()
    root_mapping: DataFrame | None = None
    embeddings: DataFrame | None = None
    try:
        walks = read_walks(spark, args.walks, args.limit_tracks)
        walk_stats = validate_walks(
            walks,
            effective_tracks,
            args.walks_per_track,
            args.max_sentence_length,
        )
        print(
            "walk_validation_passed "
            f"rows={walk_stats['rows']}, tracks={walk_stats['tracks']}, "
            f"length={walk_stats['min_walk_len']}..{walk_stats['max_walk_len']}",
        )

        root_mapping = make_root_mapping(walks)
        corpus = make_corpus(walks)
        print(
            "word2vec_training_start "
            f"vector_size={args.vector_size}, window={args.window_size}, "
            f"max_iter={args.max_iter}, step_size={args.step_size}, "
            f"partitions={args.num_partitions}, seed={args.seed}",
        )
        model = fit_word2vec(corpus, args)

        embeddings = normalize_track_vectors(model, root_mapping).persist(
            StorageLevel.DISK_ONLY,
        )
        embedding_stats = validate_embeddings(
            embeddings,
            effective_tracks,
            args.vector_size,
        )
        print(
            "embedding_validation_passed "
            f"rows={embedding_stats['rows']}, norm={embedding_stats['min_norm']:.6f}"
            f"..{embedding_stats['max_norm']:.6f}",
        )

        embeddings_path = args.output / EMBEDDINGS_NAME
        (
            embeddings.select("node_id", "track_id", "embedding")
            .repartition(args.output_partitions, "node_id")
            .sortWithinPartitions("node_id")
            .write.mode("overwrite" if args.overwrite else "error")
            .parquet(spark_path(embeddings_path))
        )
        model.write().overwrite().save(spark_path(args.output / MODEL_NAME))

        elapsed = time.monotonic() - started
        metadata = {
            "artifact": "merlin_c2_graph_embeddings",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_source": "direct_word2vec_track_token",
            "input": {
                "walks": str(args.walks.resolve()),
                **walk_stats,
            },
            "output": {
                "embeddings": EMBEDDINGS_NAME,
                "model": MODEL_NAME,
                "dtype": "float32",
                "dimension": args.vector_size,
                "l2_normalized": True,
                **embedding_stats,
            },
            "training": {
                "vector_size": args.vector_size,
                "window_size": args.window_size,
                "min_count": args.min_count,
                "max_iter": args.max_iter,
                "step_size": args.step_size,
                "num_partitions": args.num_partitions,
                "max_sentence_length": args.max_sentence_length,
                "seed": args.seed,
            },
            "spark_version": spark.version,
            "elapsed_seconds": elapsed,
        }
        write_metadata(args.output / METADATA_NAME, metadata)
        print(
            "word2vec_training_done "
            f"rows={embedding_stats['rows']}, dimension={args.vector_size}, "
            f"elapsed_seconds={elapsed:.1f}, output={args.output}",
        )
    finally:
        if embeddings is not None:
            embeddings.unpersist()
        if root_mapping is not None:
            root_mapping.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
