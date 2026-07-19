from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

FEATURES_DIR = Path(__file__).resolve().parents[2] / "src" / "features"
sys.path.insert(0, str(FEATURES_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from key_contracts import K2  # noqa: E402
from preprocessing import fit_feature_contract, transform_features  # noqa: E402
from views import build_engineered_view, build_linear_view  # noqa: E402
from test_preprocessing import row  # noqa: E402


class FeatureViewsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("YearFeatureViewsTest")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_linear_view_has_fixed_finite_vectors(self):
        data = self.spark.createDataFrame([row(index, "train") for index in range(5)])
        state = fit_feature_contract(data, K2, quantile_error=0.0)
        engineered = build_engineered_view(transform_features(data, state), state)
        vectors = build_linear_view(engineered, state).select("normalized_year", "features").collect()

        self.assertEqual(len(vectors), 5)
        for output in vectors:
            self.assertEqual(len(output["features"]), len(state["retained_columns"]))
            self.assertTrue(all(math.isfinite(value) for value in output["features"]))
            self.assertGreaterEqual(output["normalized_year"], 0.0)
            self.assertLessEqual(output["normalized_year"], 1.0)

        means = [
            sum(row_["features"][index] for row_ in vectors) / len(vectors)
            for index in range(len(state["retained_columns"]))
        ]
        self.assertTrue(all(abs(value) < 1.0e-10 for value in means))


if __name__ == "__main__":
    unittest.main()
