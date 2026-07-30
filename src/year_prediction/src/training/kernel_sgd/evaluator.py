from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from pyspark import StorageLevel
from pyspark.sql import SparkSession

from model_io import prepare_output_directory, write_json
from spark_io import write_parquet_parts

from .data import frame_points, load_frame
from .features import transform_partition
from .prediction import SCHEMA, prediction_partition, quality_metrics
from .state import load_transform


def evaluate(
    model_root: Path, input_path: Path, output_root: Path,
    spark: SparkSession, overwrite: bool = False,
) -> Path:
    started = time.perf_counter()
    model = json.loads((model_root / "model.json").read_text(encoding="ascii"))
    if model.get("model_type") != "kernel_sgd":
        raise ValueError("unsupported model type")
    transform, metadata = load_transform(model_root)
    concatenate = model["representation"] == "t90_rff"
    expected = transform.output_dimension + (transform.input_dimension if concatenate else 0)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if weights.shape != (expected,) or expected != int(model["feature_dimension"]):
        raise ValueError("model and transform dimensions differ")
    frame = load_frame(spark, input_path, ("test",)).cache()
    points = frame_points(frame).mapPartitions(
        lambda rows: transform_partition(rows, transform, concatenate)
    ).persist(StorageLevel.MEMORY_AND_DISK)
    rows = points.mapPartitions(
        lambda values: prediction_partition(values, weights, float(model["intercept"]))
    )
    predictions = spark.createDataFrame(rows, SCHEMA).cache()
    metrics = quality_metrics(predictions)
    if int(metrics["count"]) != frame.count():
        raise ValueError("prediction count differs from test input")
    output = prepare_output_directory(output_root / model["model_id"] / "test", overwrite)
    write_parquet_parts(predictions, output / "predictions.parquet")
    write_json(output / "metrics.json", {"model_id": model["model_id"], "split": "test", "metrics": metrics})
    write_json(
        output / "run_metadata.json",
        {"model_root": str(model_root.resolve()), "input": str(input_path.resolve()),
         "representation": model["representation"], "transform": metadata,
         "spark_version": spark.version, "total_seconds": time.perf_counter() - started},
    )
    predictions.unpersist(); points.unpersist(); frame.unpersist()
    return output
