from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parents[1] / "training"
ORDINAL_DIR = TRAINING_DIR / "ordinal_moe"
sys.path[:0] = [str(TRAINING_DIR), str(ORDINAL_DIR)]

from ordinal_moe_predict import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Spark Ordinal-MoE on test data")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def evaluation_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args), split="test")


def main() -> None:
    args = evaluation_args(parse_args())
    spark = SparkSession.builder.appName("YearPredictionEvaluateOrdinalMoE").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        print(json.dumps(run(args, spark), sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
