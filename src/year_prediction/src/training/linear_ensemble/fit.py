from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
LIGHTGBM_DIR = TRAINING_DIR / "lightgbm"
sys.path[:0] = [str(TRAINING_DIR), str(LIGHTGBM_DIR)]

from lightgbm_metrics import (  # noqa: E402
    add_prediction_columns,
    decade_metrics,
    regression_metrics,
)
from spark_common import parquet_inputs, prepare_output, write_json  # noqa: E402
from spark_io import write_parquet_parts  # noqa: E402

MODEL_NAMES = ("fused", "metadata", "audio")
JOIN_COLUMNS = ("track_id", "artist_id", "year", "split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a three-model linear ensemble")
    for name in MODEL_NAMES:
        parser.add_argument(f"--{name}-validation", type=Path, required=True)
        parser.add_argument(f"--{name}-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_prediction(
    spark: SparkSession, path: Path, split: str, model_name: str
) -> DataFrame:
    frame = spark.read.parquet(*parquet_inputs(path))
    required = {*JOIN_COLUMNS, "raw_prediction_year"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{model_name} {split} predictions lack {missing}")
    selected = frame.select(
        *JOIN_COLUMNS,
        F.col("raw_prediction_year").cast("double").alias(model_name),
    ).where(F.col("split") == split)
    counts = selected.agg(
        F.count("*").alias("rows"), F.countDistinct("track_id").alias("tracks")
    ).first()
    if counts is None or counts["rows"] <= 0 or counts["rows"] != counts["tracks"]:
        raise ValueError(f"{model_name} {split} predictions are empty or duplicated")
    return selected


def join_predictions(args: argparse.Namespace, spark: SparkSession, split: str):
    frames = [
        load_prediction(spark, getattr(args, f"{name}_{split}"), split, name)
        for name in MODEL_NAMES
    ]
    expected = [frame.count() for frame in frames]
    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.join(frame, list(JOIN_COLUMNS), "inner")
    joined = joined.cache()
    actual = joined.count()
    if any(count != actual for count in expected):
        raise ValueError(f"{split} prediction keys or labels do not match")
    return joined, actual


def fit_weights(validation: DataFrame) -> tuple[float, dict[str, float]]:
    rows = validation.select("year", *MODEL_NAMES).collect()
    features = np.asarray(
        [[float(row[name]) for name in MODEL_NAMES] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([float(row["year"]) for row in rows], dtype=np.float64)
    design = np.column_stack((np.ones(labels.size, dtype=np.float64), features))
    solution = np.linalg.lstsq(design, labels, rcond=None)[0]
    return float(solution[0]), {
        name: float(value) for name, value in zip(MODEL_NAMES, solution[1:])
    }


def apply_weights(
    frame: DataFrame, intercept: float, weights: dict[str, float]
) -> DataFrame:
    prediction = F.lit(intercept)
    for name in MODEL_NAMES:
        prediction = prediction + F.lit(weights[name]) * F.col(name)
    return add_prediction_columns(frame.withColumn("prediction", prediction)).select(
        *JOIN_COLUMNS,
        "raw_prediction_year",
        "clipped_prediction_year",
        "absolute_error_years",
    )


def evaluate_and_write(predictions: DataFrame, output: Path) -> dict:
    predictions = predictions.cache()
    metrics = regression_metrics(predictions)
    metrics["by_decade"] = decade_metrics(predictions)
    written = write_parquet_parts(predictions, output)
    if written != metrics["count"]:
        raise ValueError("written prediction count differs from evaluated count")
    predictions.unpersist()
    return metrics


def run(args: argparse.Namespace, spark: SparkSession) -> dict:
    prepare_output(args.output, args.overwrite)
    started = time.perf_counter()
    validation, validation_rows = join_predictions(args, spark, "validation")
    test, test_rows = join_predictions(args, spark, "test")
    intercept, weights = fit_weights(validation)
    validation_metrics = evaluate_and_write(
        apply_weights(validation, intercept, weights),
        args.output / "validation_predictions.parquet",
    )
    test_metrics = evaluate_and_write(
        apply_weights(test, intercept, weights),
        args.output / "test_predictions.parquet",
    )
    model = {
        "fit_split": "validation",
        "intercept": intercept,
        "model_type": "linear_prediction_ensemble",
        "objective": "squared_error",
        "prediction_column": "raw_prediction_year",
        "weights": weights,
    }
    metrics = {"validation": validation_metrics, "test": test_metrics}
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(args.output / "arguments.json", arguments)
    write_json(args.output / "model.json", model)
    write_json(args.output / "metrics.json", metrics)
    write_json(
        args.output / "run_metadata.json",
        {
            "application_id": spark.sparkContext.applicationId,
            "fit_rows": validation_rows,
            "model_type": model["model_type"],
            "spark_master": spark.sparkContext.master,
            "spark_version": spark.version,
            "test_rows": test_rows,
            "total_seconds": time.perf_counter() - started,
        },
    )
    validation.unpersist()
    test.unpersist()
    return metrics


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("YearPredictionLinearEnsemble").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        print(json.dumps(run(args, spark), sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
