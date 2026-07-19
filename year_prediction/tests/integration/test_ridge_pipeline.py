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

    def test_small_end_to_end_training_and_artifacts(self):
        rows = [
            ("TR0001", "AR0001", 1931, 9.0 / 89.0, [-1.0, 0.0], "train"),
            ("TR0002", "AR0002", 1940, 18.0 / 89.0, [-0.7, 0.2], "train"),
            ("TR0003", "AR0003", 1958, 36.0 / 89.0, [-0.2, -0.4], "train"),
            ("TR0004", "AR0004", 1976, 54.0 / 89.0, [0.2, 0.5], "train"),
            ("TR0005", "AR0005", 1994, 72.0 / 89.0, [0.7, -0.1], "train"),
            ("TR0006", "AR0006", 2003, 81.0 / 89.0, [1.0, 0.3], "train"),
            ("TR0007", "AR0007", 1949, 27.0 / 89.0, [-0.5, 0.1], "validation"),
            ("TR0008", "AR0008", 1985, 63.0 / 89.0, [0.5, 0.0], "validation"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "linear_vectors.parquet"
            metadata_path = root / "preprocessing_metadata.json"
            output_root = root / "models"
            frame = self.spark.createDataFrame(rows, SCHEMA)
            frame.write.partitionBy("split").parquet(input_path.resolve().as_uri())
            metadata = {
                "feature_version": "test-v1",
                "counts": {
                    "rows": 8,
                    "splits": {"train": 6, "validation": 2},
                },
                "target": {
                    "source_column": "year",
                    "output_column": "normalized_year",
                    "minimum": 1922,
                    "maximum": 2011,
                    "formula": "(year - 1922) / 89",
                },
                "outputs": {"linear_vectors": {"dimension": 2}},
            }

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
