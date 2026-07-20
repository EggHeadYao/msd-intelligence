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
RIDGE_DIR = TRAINING_DIR / "ridge"
EVALUATION_DIR = ROOT / "src" / "evaluation"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(RIDGE_DIR))

from data import load_training_data, read_training_manifest  # noqa: E402
from model_io import read_json  # noqa: E402
from target import target_contract  # noqa: E402
from train import train  # noqa: E402
from train_constants import compute_constant_baselines  # noqa: E402
from validate_ridge import validate as validate_model  # noqa: E402


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

        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "vectors.parquet"
            manifest_path = root / "manifest.json"
            output_root = root / "models"
            frame = self.spark.createDataFrame(rows, SCHEMA)
            frame.write.partitionBy("split").parquet(input_path.resolve().as_uri())
            write_manifest(manifest_path, train_count=6, validation_count=2)
            config = {
                "model_id": "ridge-integration",
                "input": str(input_path),
                "feature_manifest": str(manifest_path),
                "output_root": str(output_root),
                "objective": "ridge_squared",
                "initialization": "zero_weights_train_mean_intercept",
                "max_iterations": 6,
                "learning_rate": 0.1,
                "l2": 0.01,
                "gradient_tolerance": 0.0,
                "validation_interval": 1,
                "shuffle_partitions": 2,
                "prediction_partitions": 1,
                "execution": {
                    "aggregation": "direct_reduce",
                    "batch_fraction": 1.0,
                    "broadcast_weights": False,
                    "persist_training_data": False,
                },
            }
            output = train(config, self.spark)
            model = read_json(output / "model.json")
            history = read_json(output / "history.json")
            metrics = read_json(output / "metrics.json")
            run_metadata = read_json(output / "run_metadata.json")
            predictions = self.spark.read.parquet(
                (output / "validation_predictions.parquet").resolve().as_uri()
            )
            baselines = compute_constant_baselines(frame)
            self.assertEqual(model["feature_dimension"], DIMENSION)
            self.assertEqual(len(model["weights"]), DIMENSION)
            self.assertEqual(len(history), 6)
            self.assertLess(
                metrics["final_training_objective"],
                history[0]["training_objective_before_update"],
            )
            self.assertEqual(metrics["train"]["count"], 6)
            self.assertEqual(metrics["validation"]["count"], 2)
            self.assertEqual(predictions.count(), 2)
            self.assertEqual(baselines["mean"]["count"], 2)
            self.assertEqual(baselines["median"]["count"], 2)
            self.assertGreater(run_metadata["timing_seconds"]["gradient_reduce"], 0.0)
            validate_model(output, self.spark)

    def test_training_data_rejects_artist_overlap(self):
        rows = [
            ("TR1001", "AR1001", 1950, 28.0 / 89.0, vector(-1.0, 0.0), "train"),
            ("TR1002", "AR1001", 2000, 78.0 / 89.0, vector(1.0, 0.0), "validation"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "vectors.parquet"
            manifest_path = root / "manifest.json"
            self.spark.createDataFrame(rows, SCHEMA).write.partitionBy("split").parquet(
                input_path.resolve().as_uri()
            )
            write_manifest(manifest_path, train_count=1, validation_count=1)
            manifest = read_training_manifest(manifest_path)
            with self.assertRaisesRegex(ValueError, "Artists overlap"):
                load_training_data(self.spark, input_path, manifest)


if __name__ == "__main__":
    unittest.main()
