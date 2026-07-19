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

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "src" / "training"
EVALUATION_DIR = ROOT / "src" / "evaluation"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(EVALUATION_DIR))

from model_io import read_json  # noqa: E402
from train_constants import compute_constant_baselines  # noqa: E402
from train_sgd import train  # noqa: E402
from training_data import load_training_data  # noqa: E402


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


class RidgePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("RidgePipelineTest")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


    def test_training_data_rejects_artist_overlap(self):
        rows = [
            ("TR1001", "AR1001", 1950, 28.0 / 89.0, [-1.0, 0.0], "train"),
            ("TR1002", "AR1001", 2000, 78.0 / 89.0, [1.0, 0.0], "validation"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "linear_vectors.parquet"
            self.spark.createDataFrame(rows, SCHEMA).write.partitionBy("split").parquet(
                input_path.resolve().as_uri()
            )
            with self.assertRaisesRegex(ValueError, "artists overlap"):
                load_training_data(self.spark, input_path)


if __name__ == "__main__":
    unittest.main()
