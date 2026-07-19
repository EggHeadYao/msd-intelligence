from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent / "training"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from distributed import (  # noqa: E402
    direct_full_batch_statistics,
    evaluate_linear_model,
    prediction_row,
)
from model_io import read_json  # noqa: E402
from optimizer import gradient_norm  # noqa: E402
from train_sgd import PREDICTION_SCHEMA, ship_worker_modules  # noqa: E402
from training_data import TRAIN, VALIDATION, load_training_data, read_feature_metadata, spark_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a custom year-prediction Ridge model.")
    parser.add_argument("--model", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_finite(value: Any, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"Non-finite value at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for name, item in value.items():
            require_finite(item, f"{path}.{name}")


def require_metrics_close(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    require(int(actual["count"]) == int(expected["count"]), f"{label} count differs")
    for name in (
        "mae_years",
        "rmse_years",
        "raw_mae_years",
        "raw_rmse_years",
        "signed_error_years",
        "raw_out_of_range_rate",
    ):
        require(
            math.isclose(float(actual[name]), float(expected[name]), rel_tol=1.0e-10, abs_tol=1.0e-10),
            f"{label} {name} differs",
        )




def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("YearPredictionValidateRidgeModel").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        validate(args.model, spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
