from __future__ import annotations

import argparse
import json
import sys
from functools import reduce
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

TRAINING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAINING_DIR))

from spark_common import parquet_inputs, prepare_output, write_json  # noqa: E402
from spark_io import write_parquet_parts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge disjoint Spark OOF predictions")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, spark: SparkSession) -> dict:
    prepare_output(args.output, args.overwrite)
    frames = [spark.read.parquet(*parquet_inputs(path)) for path in args.inputs]
    columns = frames[0].columns
    if any(frame.columns != columns for frame in frames[1:]):
        raise ValueError("OOF schemas differ")
    merged = reduce(lambda left, right: left.unionByName(right), frames).cache()
    row = merged.agg(
        F.count("*").alias("rows"),
        F.countDistinct("track_id").alias("tracks"),
        F.countDistinct("artist_id").alias("artists"),
    ).first()
    if row is None or row["rows"] <= 0 or row["rows"] != row["tracks"]:
        raise ValueError("OOF track IDs are empty or duplicated")
    written = write_parquet_parts(merged, args.output / "predictions.parquet")
    metadata = {"fold_inputs": len(frames), "rows": written,
                "artists": int(row["artists"])}
    write_json(args.output / "metadata.json", metadata)
    merged.unpersist()
    return metadata


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("YearPredictionMergeOOF").getOrCreate()
    try:
        print(json.dumps(run(args, spark), sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
