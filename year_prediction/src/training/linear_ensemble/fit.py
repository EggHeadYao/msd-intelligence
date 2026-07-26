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

