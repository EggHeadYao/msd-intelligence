from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "training" / "lightgbm"
sys.path.insert(0, str(MODULE))

from lightgbm_metrics import add_prediction_columns, regression_metrics  # noqa: E402
from lightgbm_train import build_estimator  # noqa: E402


@unittest.skipUnless(
    os.environ.get("RUN_SYNAPSEML_INTEGRATION") == "1",
    "set RUN_SYNAPSEML_INTEGRATION=1 and supply SynapseML jars",
)
class SparkLightGBMPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[2]").appName(
            "SparkLightGBMPipelineTest"
        ).getOrCreate()
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_synapseml_fit_transform_and_metrics(self):
        rows = [
            (f"t{i}", f"a{i}", 1950 + i * 6, Vectors.dense(i / 5.0, (-1) ** i), i >= 8)
            for i in range(10)
        ]
        frame = self.spark.createDataFrame(
            rows, ("track_id", "artist_id", "year", "features", "is_validation")
        ).withColumnRenamed("year", "label")
        args = argparse.Namespace(
            huber_alpha=0.8, learning_rate=0.1, num_iterations=5,
            early_stopping_rounds=2, num_leaves=4, max_depth=3,
            min_data_in_leaf=1, feature_fraction=1.0, bagging_fraction=1.0,
            lambda_l2=1.0, max_bin=31, num_tasks=2, seed=472,
        )
        model = build_estimator(args, []).fit(frame)
        predictions = model.transform(frame.where("is_validation"))
        predictions = predictions.withColumn("year", predictions.label.cast("int"))
        metrics = regression_metrics(add_prediction_columns(predictions))
        self.assertEqual(metrics["count"], 2)
        self.assertGreater(len(model.getNativeModel()), 100)


if __name__ == "__main__":
    unittest.main()
