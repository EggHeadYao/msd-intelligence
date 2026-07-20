from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

MODULE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = MODULE_DIR.parent
SOURCE_DIR = EVALUATION_DIR.parent
TRAINING_DIR = SOURCE_DIR / "training"
RIDGE_DIR = TRAINING_DIR / "ridge"
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(RIDGE_DIR))

from data import EXPECTED_COLUMNS, FEATURES, SPLIT, read_training_manifest, spark_path  # noqa: E402
from distributed import prediction_row  # noqa: E402
from metrics import (  # noqa: E402
    ABSOLUTE_ERROR_COLUMN,
    add_absolute_error,
    aggregate_decade_metrics,
    aggregate_quality_metrics,
)
from model_io import (  # noqa: E402
    prepare_output_directory,
    read_json,
    sha256_file,
    write_json,
)
from target import MAX_YEAR, MIN_YEAR, YEAR_SPAN  # noqa: E402


TEST = "test"
BASE_PREDICTION_SCHEMA = StructType(
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
PREDICTION_SCHEMA = StructType(
    [
        *BASE_PREDICTION_SCHEMA.fields,
        StructField(ABSOLUTE_ERROR_COLUMN, DoubleType(), False),
    ]
)

