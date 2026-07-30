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
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            model_directory.mkdir(parents=True)
            model = {
                "format_version": 1,
                "model_id": "ridge-test",
                "model_type": "linear_ridge",
                "objective": "ridge_squared",
                "feature_dimension": DIMENSION,
                "weights": [1.0, *([0.0] * (DIMENSION - 1))],
                "intercept": 0.0,
                "l2": 0.001,
                "target": target_contract(),
                "feature_source": {
                    "input": str(vectors.resolve()),
                    "manifest": str(manifest_path.resolve()),
                    "manifest_sha256": sha256_file(manifest_path),
                    "contract_version": "year_prediction_t90_training_v1",
                    "predictor_order_sha256": "synthetic-t90-order",
                },
            }
            (model_directory / "model.json").write_text(
                json.dumps(model), encoding="ascii"
            )
            output = evaluate(
                model_directory,
                output_root,
                self.spark,
                prediction_partitions=1,
            )
            metrics = read_json(output / "metrics.json")
            decades = read_json(output / "metrics_by_decade.json")
            metadata = read_json(output / "run_metadata.json")
            predictions = self.spark.read.parquet(
                (output / "predictions.parquet").resolve().as_uri()
            )
            quality = metrics["metrics"]
            self.assertEqual(quality["count"], 3)
            self.assertEqual(quality["distinct_tracks"], 3)
            self.assertEqual(quality["distinct_artists"], 3)
            self.assertAlmostEqual(quality["mae_years"], 0.5 / 3.0)
            self.assertAlmostEqual(quality["rmse_years"], (0.25 / 3.0) ** 0.5)
            self.assertAlmostEqual(quality["median_absolute_error_years"], 0.0)
            self.assertAlmostEqual(quality["within_5_years_rate"], 1.0)
            self.assertAlmostEqual(quality["within_10_years_rate"], 1.0)
            self.assertAlmostEqual(quality["raw_out_of_range_rate"], 1.0 / 3.0)
            self.assertEqual([row["decade"] for row in decades["decades"]], [1920, 1960, 2010])
            self.assertAlmostEqual(quality["macro_decade_mae_years"], 0.5 / 3.0)
            self.assertEqual(predictions.count(), 3)
            self.assertIn("absolute_error_years", predictions.columns)
            self.assertEqual(metadata["evaluation"], "course_test_benchmark")
            self.assertEqual(metadata["counts"], {"tracks": 3, "artists": 3})


if __name__ == "__main__":
    unittest.main()
