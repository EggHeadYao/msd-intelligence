from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from lightgbm_metrics import add_prediction_columns, regression_metrics  # noqa: E402
from lightgbm_train import build_estimator  # noqa: E402
from spark_common import (  # noqa: E402
    load_feature_contract,
    load_feature_frame,
    prepare_output,
    write_json,
)

STAGES = ("capacity", "regularization", "learning_rate", "huber")
DEFAULTS: dict[str, Any] = {
    "huber_alpha": 0.9,
    "learning_rate": 0.04,
    "num_leaves": 12,
    "max_depth": 6,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "lambda_l2": 5.0,
    "max_bin": 255,
    "num_iterations": 5000,
    "early_stopping_rounds": 200,
    "seed": 472,
}
GRIDS: dict[str, tuple[dict[str, Any], ...]] = {
    "capacity": (
        {"num_leaves": 12, "max_depth": 6, "min_data_in_leaf": 500},
        {"num_leaves": 24, "max_depth": 8, "min_data_in_leaf": 300},
        {"num_leaves": 31, "max_depth": 8, "min_data_in_leaf": 200},
    ),
    "regularization": (
        {"feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l2": 5.0},
        {"feature_fraction": 1.0, "bagging_fraction": 0.85, "lambda_l2": 20.0},
        {"feature_fraction": 0.75, "bagging_fraction": 0.75, "lambda_l2": 10.0},
    ),
    "learning_rate": (
        {"learning_rate": 0.08, "num_iterations": 2500},
        {"learning_rate": 0.04, "num_iterations": 5000},
        {"learning_rate": 0.02, "num_iterations": 8000},
    ),
    "huber": (
        {"huber_alpha": 0.8},
        {"huber_alpha": 0.9},
        {"huber_alpha": 0.95},
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune distributed Spark LightGBM")
    parser.add_argument(
        "--input",
