"""MERLIN C2: meta-path random walk generation (main entry point).

Usage:
  spark-submit --driver-memory 8g merlin/embedding/graph/main.py \
    --input parquets/prepared --output parquets/walks
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from merlin.embedding.graph.config import NUM_WALKS, SEED, WALK_LENGTH
from merlin.embedding.graph.index import load_and_build_index
from merlin.embedding.graph.walk import generate_walks_for_partition


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


def main() -> None:
    args = parse_args()
    spark: SparkSession = _make_spark()

    input_dir: str = args.input
    output_dir: str = args.output
    tmp_dir: str = args.tmp_dir
    r: int = args.walks
    target_len: int = args.length
    sample: int = args.sample

    if Path(tmp_dir.replace("file://", "")).exists():
        shutil.rmtree(tmp_dir.replace("file://", ""))

    # --- Phase A: Build adjacency index ---
    node_to_int, int_to_node, int_to_type = load_and_build_index(
        spark,
        f"{input_dir}/graph_edges.parquet",
        tmp_dir,
    )

    spark.stop()


if __name__ == "__main__":
    main()
