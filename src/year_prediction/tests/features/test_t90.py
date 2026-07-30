from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "src" / "features"
sys.path.insert(0, str(FEATURE_DIR))

from contract import ARTIST_ID, AUDIT_COLUMNS, SPLIT, T90_COLUMNS, TRACK_ID, YEAR  # noqa: E402
from t90 import build_t90  # noqa: E402


class T90ProjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("YearPredictionT90Test")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_projection_preserves_values_and_order(self):
        schema = ", ".join(
            (
                f"{TRACK_ID} string",
                f"{ARTIST_ID} string",
                f"{YEAR} int",
                f"{SPLIT} string",
                *(f"{column} double" for column in T90_COLUMNS),
                "unused double",
            )
        )
        values = ("track", "artist", 1990, "train", *map(float, range(90)), -1.0)
        result = build_t90(self.spark.createDataFrame([values], schema=schema))
        self.assertEqual(tuple(result.columns), AUDIT_COLUMNS + T90_COLUMNS)
        row = result.first()
        self.assertEqual(row[T90_COLUMNS[0]], 0.0)
        self.assertEqual(row[T90_COLUMNS[-1]], 89.0)


if __name__ == "__main__":
    unittest.main()
