from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "training" / "ordinal_moe"
sys.path.insert(0, str(MODULE))

from ordinal_moe_core import (  # noqa: E402
    LossConfig,
    ParameterLayout,
    distributed_gradient,
    initialize_parameters,
    prediction_partition,
)


class SparkOrdinalMoEPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[2]").appName(
            "SparkOrdinalMoEPipelineTest"
        ).getOrCreate()
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_distributed_gradient_and_prediction(self):
        points = self.spark.sparkContext.parallelize([
            ("t1", "a1", 1960, np.array([-1.0, 0.0])),
            ("t2", "a2", 1980, np.array([-0.2, 0.5])),
            ("t3", "a3", 2000, np.array([0.7, -0.1])),
            ("t4", "a4", 2010, np.array([1.0, 0.2])),
        ], 2)
        layout = ParameterLayout(2)
        histogram = [0] * 90
        for year in (1960, 1980, 2000, 2010):
            histogram[year - 1922] += 1
        parameters = initialize_parameters(layout, histogram, 472)
        gradient, losses, count = distributed_gradient(
            points, parameters, layout, LossConfig(), 1.0e-4, 2
        )
        self.assertEqual(count, 4)
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertTrue(np.isfinite(losses["total"]))
        predictions = points.mapPartitions(
            lambda rows: prediction_partition(
                rows, parameters, layout, LossConfig(), 2
            )
        ).collect()
        self.assertEqual(len(predictions), 4)
        self.assertTrue(all(1922.0 <= row[6] <= 2011.0 for row in predictions))


if __name__ == "__main__":
    unittest.main()
