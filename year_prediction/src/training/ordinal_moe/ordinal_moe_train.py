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
