from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from model_io import prepare_output_directory, write_json
from spark_io import write_parquet_parts

from .data import frame_points, load_frame
from .features import transform_partition
from .fit import fit
from .prediction import SCHEMA, prediction_partition, quality_metrics
from .state import fit_transform, save_transform


def train(config: dict, spark: SparkSession, overwrite: bool = False) -> Path:
    started = time.perf_counter()
    output = prepare_output_directory(Path(config["output_root"]) / config["model_id"], overwrite)
    frame = load_frame(spark, Path(config["input"]), ("train", "validation")).cache()
    counts = {row["split"]: int(row["count"]) for row in frame.groupBy("split").count().collect()}
    if set(counts) != {"train", "validation"}:
        raise ValueError("train and validation splits are required")
    transform, transform_metadata = fit_transform(frame, config)
    concatenate = config["representation"] == "t90_rff"
    training = frame_points(frame.where(F.col("split") == "train")).mapPartitions(
        lambda rows: transform_partition(rows, transform, concatenate)
    ).persist(StorageLevel.DISK_ONLY)
    validation = frame_points(frame.where(F.col("split") == "validation")).mapPartitions(
        lambda rows: transform_partition(rows, transform, concatenate)
    ).persist(StorageLevel.DISK_ONLY)
    train_mean = float(frame.where(F.col("split") == "train").agg(F.avg("normalized_year")).first()[0])
    if training.count() != counts["train"] or validation.count() != counts["validation"]:
        raise ValueError("transformed split counts differ from input")
    frame.unpersist()
    dimension = transform.output_dimension + (transform.input_dimension if concatenate else 0)
    result = fit(training, validation, dimension, train_mean, config)
    weights = np.asarray(result["weights"], dtype=np.float64)
    prediction_rows = validation.mapPartitions(
        lambda rows: prediction_partition(rows, weights, float(result["intercept"]))
    )
    predictions = spark.createDataFrame(prediction_rows, SCHEMA).cache()
    metrics = quality_metrics(predictions)
    write_parquet_parts(predictions, output / "validation_predictions.parquet")
    save_transform(output, transform, transform_metadata)
    write_json(output / "history.json", result["history"])
    write_json(output / "metrics.json", {"validation": metrics, "best_iteration": result["best_iteration"]})
    write_json(
        output / "model.json",
        {
            "format_version": 1, "model_id": config["model_id"], "model_type": "kernel_sgd",
            "representation": config["representation"], "loss": config["loss"],
            "feature_dimension": dimension, "weights": weights.tolist(),
            "intercept": float(result["intercept"]), "l2": float(config["l2"]),
            "target": {"minimum": 1922, "maximum": 2011, "span": 89},
        },
    )
    write_json(
        output / "run_metadata.json",
        {"configuration": config, "counts": counts, "spark_version": spark.version,
         "spark_master": spark.sparkContext.master, "total_seconds": time.perf_counter() - started},
    )
    predictions.unpersist(); training.unpersist(); validation.unpersist()
    return output
