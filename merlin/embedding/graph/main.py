"""MERLIN C2: meta-path random walk generation (main entry point).

Usage:
  spark-submit --driver-memory 8g merlin/embedding/graph/main.py \
    --input parquets/prepared --output parquets/walks
"""

from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession

from merlin.embedding.graph.config import NUM_WALKS, SEED, WALK_LENGTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to prepared Parquet directory (contains graph_edges.parquet, songs_metadata.parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for walk_sequences.parquet",
    )
    parser.add_argument(
        "--walks",
        type=int,
        default=NUM_WALKS,
        help=f"Walks per song (default: {NUM_WALKS})",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=WALK_LENGTH,
        help=f"Target song nodes per walk (default: {WALK_LENGTH})",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Dev mode: only process first N songs (0 = full 1M)",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="/tmp/c2_index",
        help="Temp directory for intermediate adjacency Parquet files",
    )
    return parser.parse_args()


def _make_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinC2Walk")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "500")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )


if __name__ == "__main__":
    main()
