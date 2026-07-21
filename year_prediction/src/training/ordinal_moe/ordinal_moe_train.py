from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
LIGHTGBM_DIR = TRAINING_DIR / "lightgbm"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(LIGHTGBM_DIR))

from lightgbm_metrics import add_prediction_columns, decade_metrics, regression_metrics  # noqa: E402
from ordinal_moe_core import (  # noqa: E402
    MAX_YEAR,
    MIN_YEAR,
    LossConfig,
    ParameterLayout,
    adam_step,
    distributed_gradient,
    initialize_parameters,
    prediction_partition,
)
from spark_io import write_parquet_parts  # noqa: E402
from spark_common import (  # noqa: E402
    assert_artist_isolation,
    fit_standardization,
    load_feature_contract,
    load_feature_frame,
    point_rdd,
    parquet_inputs,
    prepare_output,
    split_counts,
    standardize_frame,
    write_json,
)

PREDICTION_SCHEMA = StructType(
    [
        StructField("track_id", StringType(), False),
        StructField("artist_id", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("ordinal_year", DoubleType(), False),
        StructField("moe_year", DoubleType(), False),
        StructField("direct_year", DoubleType(), False),
        StructField("blend_year", DoubleType(), False),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train distributed Spark Ordinal-MoE")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("parquets/year_prediction/features/full_tabular.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("parquets/year_prediction/features/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parquets/year_prediction/models/ordinal-moe-spark-v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--fold-assignments", type=Path)
    parser.add_argument("--validation-fold", type=int)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--l2", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--loss-ordinal", type=float, default=0.35)
    parser.add_argument("--loss-moe", type=float, default=0.45)
    parser.add_argument("--loss-direct", type=float, default=0.05)
    parser.add_argument("--loss-decade", type=float, default=0.12)
    parser.add_argument("--loss-consistency", type=float, default=0.03)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--expert-span", type=float, default=8.0)
    parser.add_argument("--blend-ordinal", type=float, default=0.20)
    parser.add_argument("--blend-moe", type=float, default=0.80)
    parser.add_argument("--blend-direct", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=472)
    return parser.parse_args()


def loss_config(args: argparse.Namespace) -> LossConfig:
    config = LossConfig(
        ordinal=args.loss_ordinal,
        moe=args.loss_moe,
        direct=args.loss_direct,
        decade=args.loss_decade,
        consistency=args.loss_consistency,
        huber_delta=args.huber_delta,
        expert_span=args.expert_span,
        blend_ordinal=args.blend_ordinal,
        blend_moe=args.blend_moe,
        blend_direct=args.blend_direct,
    )
    config.validate()
    return config


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("training counts must be positive")
    if args.partitions <= 0:
        raise ValueError("partitions must be positive")
    if not 0.0 < args.sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1]")
    if args.learning_rate <= 0.0 or args.l2 < 0.0:
        raise ValueError("optimizer values are invalid")
    if (args.fold_assignments is None) != (args.validation_fold is None):
        raise ValueError("fold assignments and validation fold must be set together")
    if args.fold_assignments is not None and args.evaluate_test:
        raise ValueError("OOF training cannot evaluate the test split")


def year_histogram(train: DataFrame) -> list[int]:
    result = [0] * (int(MAX_YEAR - MIN_YEAR) + 1)
    for row in train.groupBy("year").count().collect():
        result[int(row["year"] - MIN_YEAR)] = int(row["count"])
    return result


def prediction_rdd(points, parameters, layout, config, batch_size):
    broadcast = points.context.broadcast(parameters)
    try:
        return points.mapPartitions(
            lambda rows: prediction_partition(
                rows, broadcast.value, layout, config, batch_size
            )
        ).collect()
    finally:
        broadcast.destroy()


def validation_mae(points, parameters, layout, config, batch_size) -> float:
    broadcast = points.context.broadcast(parameters)
    try:
        absolute, count = points.mapPartitions(
            lambda rows: prediction_partition(
                rows, broadcast.value, layout, config, batch_size
            )
        ).map(lambda row: (abs(row[6] - row[2]), 1)).treeReduce(
            lambda left, right: (left[0] + right[0], left[1] + right[1])
        )
    finally:
        broadcast.destroy()
    return float(absolute / count)


def prediction_frame(
    spark: SparkSession, points, parameters, layout, config, batch_size
) -> DataFrame:
    values = points.mapPartitions(
        lambda rows: prediction_partition(
            rows, parameters, layout, config, batch_size
        )
    )
    return spark.createDataFrame(values, PREDICTION_SCHEMA)


def evaluate_heads(frame: DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head in ("ordinal", "moe", "direct", "blend"):
        predictions = add_prediction_columns(frame, f"{head}_year")
        metrics = regression_metrics(predictions)
        metrics["by_decade"] = decade_metrics(predictions)
        result[head] = metrics
    return result


def run(args: argparse.Namespace, spark: SparkSession) -> dict[str, Any]:
    validate_args(args)
    config = loss_config(args)
    prepare_output(args.output, args.overwrite)
    started = time.perf_counter()
    contract = load_feature_contract(args.manifest)
    requested_splits = (("train",) if args.fold_assignments is not None else
                        (("train", "validation", "test") if args.evaluate_test
                         else ("train", "validation")))
    raw = load_feature_frame(
        spark, args.input, contract, args.max_rows_per_split, requested_splits
    )
    if args.fold_assignments is not None:
        folds = spark.read.parquet(*parquet_inputs(args.fold_assignments)).select(
            "artist_id", "fold"
        )
        raw = raw.join(folds, "artist_id", "inner").withColumn(
