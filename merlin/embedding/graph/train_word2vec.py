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


        )
        w2v_model = word2vec.fit(walks_df)

        # Get Word2Vec vectors as dict (node_id -> vector)
        vec_dict = {}
        for row in w2v_model.getVectors().collect():
            vec_dict[row.word] = [float(x) for x in row.vector.toArray()]

        # Broadcast the vector dictionary
        bc_vec_dict = spark.sparkContext.broadcast(vec_dict)

        # For each song, average the node vectors from all walks
        def avg_embeddings(track_id, walk_strings_list):
            """Average embeddings for all nodes in all walks for a given track."""
            vec_dict = bc_vec_dict.value
            all_vecs = []
            
            for walk_strings in walk_strings_list:
                for node_id in walk_strings:
                    if node_id in vec_dict:
                        all_vecs.append(vec_dict[node_id])
            
            if not all_vecs:
                # If no vectors found, return zeros
                return [0.0] * len(next(iter(vec_dict.values())))
            
            # Compute average
            dim = len(all_vecs[0])
            avg_vec = [sum(v[i] for v in all_vecs) / len(all_vecs) for i in range(dim)]
            
            # Normalize to unit length
            norm_sq = sum(x * x for x in avg_vec)
            if norm_sq == 0:
                return [0.0] * dim
            norm = norm_sq ** 0.5
            return [x / norm for x in avg_vec]

        # Group walks by track_id and compute average embeddings
        song_vecs_rdd = (
            walks_df.select("track_id", "walk_strings")
            .rdd.map(lambda row: (row.track_id, row.walk_strings))
            .groupByKey()
            .map(lambda x: (x[0], avg_embeddings(x[0], list(x[1]))))
        )

        # Convert back to DataFrame
        embeddings = spark.createDataFrame(
            song_vecs_rdd,
            schema="track_id string, embedding array<double>"
        )

        # Write outputs
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        embeddings_path = str(output_path / "song_embeddings_graph.parquet")
        embeddings.write.mode("overwrite").parquet(embeddings_path)
        print(f"[OK] Saved embeddings to {embeddings_path}")

        # Save Word2Vec model
        model_path = str(output_path / "word2vec_model")
        w2v_model.write().overwrite().save(model_path)
        print(f"[OK] Saved Word2Vec model to {model_path}")

        # Count results
        song_count = embeddings.count()
        vocab_size = len(vec_dict)

        # Write metadata
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": args.input,
            "model": "Word2Vec (Skip-gram, normalized)",
            "vector_size": args.vector_size,
            "window_size": args.window_size,
            "num_iterations": args.num_iterations,
            "seed": args.seed,
            "songs_with_embeddings": song_count,
            "vocabulary_size": vocab_size,
            "embedding_format": "array<double> (normalized to unit length)",
        }

        metadata_path = output_path / "graph_encoder_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"[OK] Saved metadata to {metadata_path}")

        print(
            f"\n[OK] Word2Vec training complete: {song_count} songs, "
            f"{vocab_size} vocabulary size, embedding_dim={args.vector_size}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
