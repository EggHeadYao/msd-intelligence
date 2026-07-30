from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession

from .runner_impl import train


ALGORITHMS = {
    "pca_ridge": ("pca", "squared"),
    "rff_ridge": ("t90_rff", "squared"),
    "rff_huber": ("rff", "huber"),
}


def read_config(path: Path, algorithm: str) -> dict:
    config = json.loads(path.read_text(encoding="ascii"))
    representation, loss = ALGORITHMS[algorithm]
    if config.get("algorithm") != algorithm:
        raise ValueError("configuration algorithm differs from entry point")
    config["representation"] = representation
    config["loss"] = loss
    required = (
        "model_id", "input", "output_root", "max_iterations", "learning_rate",
        "l2", "gradient_tolerance", "validation_interval",
        "early_stopping_patience", "early_stopping_min_delta",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing configuration fields: {missing}")
    return config


def main(algorithm: str) -> None:
    parser = argparse.ArgumentParser(description=f"Train the {algorithm} year model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config, algorithm)
    spark = SparkSession.builder.appName(f"YearPrediction-{algorithm}").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        output = train(config, spark, args.overwrite)
        print(json.dumps({"output": str(output)}, sort_keys=True))
    finally:
        spark.stop()
