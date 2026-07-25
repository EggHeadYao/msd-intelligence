from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession

from .run import evaluate


def main(expected_representation: str) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a kernel SGD model on test data")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    model = json.loads((args.model_root / "model.json").read_text(encoding="ascii"))
    if model.get("representation") != expected_representation:
        raise ValueError("model representation differs from evaluator")
    spark = SparkSession.builder.appName("YearPredictionKernelSgdTest").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        result = evaluate(args.model_root, args.input, args.output, spark, args.overwrite)
        print(json.dumps({"output": str(result)}, sort_keys=True))
    finally:
        spark.stop()
