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

    # --- Phase B: Load songs & generate walks ---
    songs_df: DataFrame = spark.read.parquet(
        f"{input_dir}/songs_metadata.parquet",
    ).select("track_id")
    if sample > 0:
        songs_df = songs_df.limit(sample)
    song_count: int = songs_df.count()
    print(f"Generating walks for {song_count} songs (r={r}, L={target_len})")

    bc_vocab = spark.sparkContext.broadcast(node_to_int)
    bc_tmp = spark.sparkContext.broadcast(tmp_dir)

    def _run_partition(iterator):
        return generate_walks_for_partition(
            iterator,
            bc_tmp.value,
            bc_vocab.value,
            r,
            target_len,
            SEED,
        )

    walk_schema = StructType(
        [
            StructField("track_id", StringType(), False),
            StructField("walk_id", IntegerType(), False),
            StructField("path_name", StringType(), False),
            StructField("walk_seq", ArrayType(IntegerType()), False),
            StructField("walk_len", IntegerType(), False),
        ]
    )

    walks: DataFrame = spark.createDataFrame(
        songs_df.rdd.mapPartitions(_run_partition),
        schema=walk_schema,
    )

    out_path: str = f"{output_dir}/walk_sequences.parquet"
    walks.write.mode("overwrite").parquet(out_path)

    total: int = walks.count()
    distinct: int = walks.select("track_id").distinct().count()
    print(
        f"Done: {total} walks for {distinct} songs saved to {out_path}",
    )

    shutil.rmtree(tmp_dir.replace("file://", ""))

    spark.stop()


if __name__ == "__main__":
    main()
