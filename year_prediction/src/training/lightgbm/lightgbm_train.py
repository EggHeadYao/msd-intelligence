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
