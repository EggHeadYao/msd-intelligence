from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = ROOT / "src" / "training"
RIDGE_DIR = TRAINING_DIR / "ridge"
EVALUATION_DIR = ROOT / "src" / "evaluation"
RIDGE_EVALUATION_DIR = EVALUATION_DIR / "ridge"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(RIDGE_DIR))
sys.path.insert(0, str(RIDGE_EVALUATION_DIR))

from evaluate import evaluate  # noqa: E402
from model_io import read_json, sha256_file  # noqa: E402
from target import target_contract  # noqa: E402


DIMENSION = 90
SCHEMA = StructType(
    [
        StructField("track_id", StringType(), False),
        StructField("artist_id", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("normalized_year", DoubleType(), False),
        StructField("features", ArrayType(DoubleType(), False), False),
        StructField("split", StringType(), False),
    ]
)


def vector(first: float) -> list[float]:
    return [first, *([0.0] * (DIMENSION - 1))]

