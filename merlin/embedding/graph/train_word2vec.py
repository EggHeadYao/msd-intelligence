"""MERLIN C2: Word2Vec training on meta-path walk sequences (RDD-based aggregation)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pyspark.ml.feature import Word2Vec
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to walk_sequences.parquet from walk generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for song_embeddings_graph.parquet and metadata",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=128,
        help="Word2Vec embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Word2Vec context window size (default: 5)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="Training iterations (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def _make_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinC2Word2Vec")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "500")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )


def main() -> None:
    args = parse_args()
    spark: SparkSession = _make_spark()
    spark.sparkContext.setLogLevel("WARN")

    try:
        # Read walk sequences
        print(f"Reading walk sequences from {args.input}/walk_sequences.parquet")
        walks_df: DataFrame = spark.read.parquet(
            f"{args.input}/walk_sequences.parquet"
        )

        # Convert walk_seq (Array[Int]) to walk_strings (Array[String])
        def cast_to_string_array(arr):
            if arr is None:
                return None
            return [f"node_{i}" for i in arr]
        
        cast_to_string_udf = F.udf(cast_to_string_array, F.ArrayType(F.StringType()))
        walks_df = walks_df.withColumn(
            "walk_strings",
            cast_to_string_udf(F.col("walk_seq")),
        ).select("track_id", "walk_strings")

        print(f"Training Word2Vec with vector_size={args.vector_size}, "
              f"window_size={args.window_size}, num_iterations={args.num_iterations}")

        # Train Word2Vec
        word2vec = Word2Vec(
            vectorSize=args.vector_size,
            windowSize=args.window_size,
            minCount=1,
            maxIter=args.num_iterations,
            seed=args.seed,
            inputCol="walk_strings",
            outputCol="w2v_vector",
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
        print(f"✓ Saved embeddings to {embeddings_path}")

        # Save Word2Vec model
        model_path = str(output_path / "word2vec_model")
        w2v_model.write().overwrite().save(model_path)
        print(f"✓ Saved Word2Vec model to {model_path}")

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
        print(f"✓ Saved metadata to {metadata_path}")

        print(
            f"\n✓ Word2Vec training complete: {song_count} songs, "
            f"{vocab_size} vocabulary size, embedding_dim={args.vector_size}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
