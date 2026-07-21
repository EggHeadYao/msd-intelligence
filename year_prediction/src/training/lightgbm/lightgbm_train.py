from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from synapse.ml.lightgbm import LightGBMRegressor

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from lightgbm_metrics import (  # noqa: E402
    add_prediction_columns,
    constant_baselines,
    decade_metrics,
    regression_metrics,
)
from spark_io import write_native_model, write_parquet_parts  # noqa: E402
from spark_common import (  # noqa: E402
    assert_artist_isolation,
    load_feature_contract,
    load_feature_frame,
    prepare_output,
    split_counts,
    write_json,
)

DEFAULT_INPUT = Path("parquets/year_prediction/features/full_tabular.parquet")
DEFAULT_MANIFEST = Path("parquets/year_prediction/features/manifest.json")
DEFAULT_OUTPUT = Path("parquets/year_prediction/models/lightgbm-spark-v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train distributed SynapseML LightGBM")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--feature-fraction", type=float, default=1.0)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--lambda-l2", type=float, default=20.0)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--huber-alpha", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=472)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_iterations <= 0 or args.early_stopping_rounds < 0:
        raise ValueError("iteration counts are invalid")
    if args.num_tasks < 0:
        raise ValueError("num_tasks cannot be negative")
    if not 0.0 < args.learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    for name in ("feature_fraction", "bagging_fraction", "huber_alpha"):
        if not 0.0 < float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")


def build_estimator(args: argparse.Namespace, categorical: list[int]) -> LightGBMRegressor:
    return LightGBMRegressor(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        validationIndicatorCol="is_validation",
        objective="huber",
        metric="l1",
        alpha=args.huber_alpha,
        learningRate=args.learning_rate,
        numIterations=args.num_iterations,
        earlyStoppingRound=args.early_stopping_rounds,
        numLeaves=args.num_leaves,
        maxDepth=args.max_depth,
        minDataInLeaf=args.min_data_in_leaf,
        featureFraction=args.feature_fraction,
        baggingFraction=args.bagging_fraction,
        baggingFreq=1,
        lambdaL2=args.lambda_l2,
        maxBin=args.max_bin,
        categoricalSlotIndexes=categorical,
        useMissing=True,
        zeroAsMissing=False,
        dataTransferMode="streaming",
        useSingleDatasetMode=True,
        numTasks=args.num_tasks,
        seed=args.seed,
        deterministic=True,
        verbosity=1,
    )


def prediction_artifact(model: Any, frame: DataFrame) -> DataFrame:
    transformed = model.transform(frame)
    return add_prediction_columns(transformed).select(
        "track_id",
        "artist_id",
        "year",
        "split",
        "raw_prediction_year",
        "clipped_prediction_year",
        "absolute_error_years",
    )


def evaluate_split(model: Any, frame: DataFrame, output: Path, split: str) -> dict[str, Any]:
    predictions = prediction_artifact(model, frame.where(F.col("split") == split)).cache()
    metrics = regression_metrics(predictions)
    metrics["by_decade"] = decade_metrics(predictions)
    write_parquet_parts(
        predictions,
        output / f"{split}_predictions.parquet",
        (
            "track_id",
            "artist_id",
            "year",
            "split",
            "raw_prediction_year",
            "clipped_prediction_year",
            "absolute_error_years",
        ),
    )
    predictions.unpersist()
    return metrics

