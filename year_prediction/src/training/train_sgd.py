from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

MODULE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = MODULE_DIR.parent / "evaluation"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(EVALUATION_DIR))

from distributed import (  # noqa: E402
    direct_full_batch_statistics,
    evaluate_linear_model,
    prediction_row,
)
from model_io import (  # noqa: E402
    prepare_output_directory,
    read_json,
    sha256_file,
    write_json,
)
from optimizer import gradient_norm, gradient_step  # noqa: E402
from training_data import (  # noqa: E402
    TRAIN,
    VALIDATION,
    load_training_data,
    read_feature_metadata,
    spark_path,
)


PREDICTION_SCHEMA = StructType(
    [
        StructField("track_id", StringType(), False),
        StructField("artist_id", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("normalized_year", DoubleType(), False),
        StructField("normalized_prediction", DoubleType(), False),
        StructField("raw_prediction_year", DoubleType(), False),
        StructField("clipped_prediction_year", DoubleType(), False),
    ]
)


def default_config_path() -> Path:
    return MODULE_DIR.parents[1] / "config" / "experiment_b" / "d0_direct.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configurable Spark SGD model.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--model-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_number(config: dict[str, Any], name: str, minimum: float, strict: bool) -> float:
    value = float(config[name])
    valid = value > minimum if strict else value >= minimum
    if not valid or not math.isfinite(value):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return value


