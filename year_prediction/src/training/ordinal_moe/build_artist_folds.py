from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

TRAINING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAINING_DIR))

from spark_common import (  # noqa: E402
    load_feature_contract,
    parquet_inputs,
    prepare_output,
    write_json,
)
from spark_io import write_parquet_parts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build artist-isolated Spark folds")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=472)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, spark: SparkSession) -> dict:
    if args.folds < 2:
        raise ValueError("fold count must be at least two")
    prepare_output(args.output, args.overwrite)
    load_feature_contract(args.manifest)
    tracks = spark.read.parquet(*parquet_inputs(args.input)).select(
        F.col("track_id").cast("string"),
        F.col("artist_id").cast("string"),
        F.col("year").cast("int"),
        F.col("split").cast("string"),
    ).where(F.col("split") == "train")
    artists = tracks.groupBy("artist_id").agg(
        F.avg("year").alias("mean_year"), F.count("*").alias("track_count")
    ).withColumn("decade", (F.floor(F.col("mean_year") / 10) * 10).cast("int"))
    order = Window.partitionBy("decade").orderBy(
        F.xxhash64("artist_id", F.lit(args.seed)), "artist_id"
    )
    assignments = artists.withColumn(
        "fold", F.pmod(F.row_number().over(order) - 1, F.lit(args.folds)).cast("int")
    ).select("artist_id", "fold", "decade", "track_count")
    written = write_parquet_parts(assignments, args.output / "artist_folds.parquet")
    fold_counts = [row.asDict() for row in assignments.groupBy("fold").agg(
        F.count("*").alias("artists"), F.sum("track_count").alias("tracks")
    ).orderBy("fold").collect()]
    metadata = {"folds": args.folds, "seed": args.seed, "artists": written,
                "fold_counts": fold_counts}
    write_json(args.output / "metadata.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("YearPredictionArtistFolds").getOrCreate()
    try:
        print(json.dumps(run(args, spark), sort_keys=True))
    finally:
        spark.stop()
