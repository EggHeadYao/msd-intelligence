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


class RidgeTestEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("RidgeTestEvaluationTest")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_complete_test_evaluation_outputs(self):
        rows = [
            ("TR0001", "AR0001", 1922, 0.0, vector(0.0), "test"),
            ("TR0002", "AR0002", 1967, 45.0 / 89.0, vector(0.5), "test"),
            ("TR0003", "AR0003", 2011, 1.0, vector(1.1), "test"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vectors = root / "vectors.parquet"
            manifest_path = root / "manifest.json"
            model_directory = root / "models" / "ridge-test"
            output_root = root / "results"
            self.spark.createDataFrame(rows, SCHEMA).write.partitionBy("split").parquet(
                vectors.resolve().as_uri()
            )
            manifest = {
                "format_version": 1,
                "contract_version": "year_prediction_t90_training_v1",
                "source": {"predictor_order_sha256": "synthetic-t90-order"},
                "target": target_contract(),
                "preprocessing": {
                    "fit_split": "train",
                    "dimension": DIMENSION,
                    "features": [
                        {"name": f"t90_{index}"} for index in range(DIMENSION)
                    ],
                },
                "counts": {
                    "splits": {
                        "train": {"tracks": 1, "artists": 1},
                        "validation": {"tracks": 1, "artists": 1},
                        "test": {"tracks": 3, "artists": 3},
                    }
                },
                "output": {
                    "path": "vectors.parquet",
                    "partition_column": "split",
                    "columns": [
                        "track_id",
                        "artist_id",
                        "year",
                        "normalized_year",
                        "features",
                        "split",
                    ],
                },
            }
