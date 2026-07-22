import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

TRAINING = Path(__file__).resolve().parents[3] / "src" / "training"
sys.path.insert(0, str(TRAINING))

from kernel_sgd.prediction import SCHEMA, quality_metrics  # noqa: E402


class KernelSgdEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[2]").appName("KernelSgdEvaluationTest").getOrCreate()
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_clipped_year_metrics(self):
        rows = [
            ("TR1", "AR1", 1922, 0.0, -0.1, 1913.1, 1922.0, 0.0),
            ("TR2", "AR2", 1960, 38.0 / 89.0, 43.0 / 89.0, 1965.0, 1965.0, 5.0),
            ("TR3", "AR3", 2011, 1.0, 1.2, 2028.8, 2011.0, 0.0),
        ]
        metrics = quality_metrics(self.spark.createDataFrame(rows, SCHEMA))
        self.assertEqual(metrics["count"], 3)
        self.assertAlmostEqual(metrics["mae_years"], 5.0 / 3.0)
        self.assertAlmostEqual(metrics["within_5_years_rate"], 1.0)
        self.assertAlmostEqual(metrics["raw_out_of_range_rate"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
