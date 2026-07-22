import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "training"), str(Path(__file__).resolve().parent)]

from kernel_sgd_case import exercise  # noqa: E402


class RffRidgePipelineTest(unittest.TestCase):
    def test_train_validate_and_test(self):
        spark = SparkSession.builder.master("local[2]").appName("RffRidgePipelineTest").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        try:
            exercise(spark, "t90_rff", "squared")
        finally:
            spark.stop()


if __name__ == "__main__":
    unittest.main()
