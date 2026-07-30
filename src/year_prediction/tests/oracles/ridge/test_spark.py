from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession


ORACLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ORACLE_DIR))

from spark_oracle import load_fixture, run_oracle  # noqa: E402


class GradientOracleIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("RidgeGradientOracleTest")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("WARN")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_spark_matches_reference_for_each_partition_count(self) -> None:
        fixture = load_fixture(ORACLE_DIR / "fixture.json")
        result = run_oracle(self.spark.sparkContext, fixture)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["spark_partitions_checked"], [1, 2, 4])
        self.assertLess(
            result["finite_difference_relative_error"],
            fixture["finite_difference_tolerance"],
        )


if __name__ == "__main__":
    unittest.main()
