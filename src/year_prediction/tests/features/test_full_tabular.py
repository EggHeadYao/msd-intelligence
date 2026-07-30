from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "src" / "features"
sys.path.insert(0, str(FEATURE_DIR))

from contract import (  # noqa: E402
    AUDIT_COLUMNS,
    DERIVED_SCALAR_COLUMNS,
    GLOBAL_SCALAR_COLUMNS,
    TRACK_ID,
)
from full_tabular import build_full_tabular  # noqa: E402


class FullTabularProjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("YearPredictionFullTabularTest")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def make_frame(self):
        schema = (
            "track_id string, artist_id string, year int, split string, shared double, "
            "loudness double, tempo double, duration double, key int, key_confidence double, "
            "mode int, mode_confidence double, time_signature int, "
            "time_signature_confidence double, end_of_fade_in double, "
            "start_of_fade_out double"
        )
        rows = [
            ("valid", "artist", 2000, "train", 7.0, -5.0, 120.0, 100.0, 2, 0.8, 1, 0.9, 4, 0.7, 1.0, 90.0),
            (
                "tolerated",
                "artist",
                2000,
                "train",
                8.0,
                -6.0,
                100.0,
                100.0,
                3,
                0.7,
                0,
                0.8,
                3,
                0.6,
                1.0,
                100.0005,
            ),
            ("invalid", "artist", None, None, 9.0, -7.0, 0.0, 100.0, -1, 0.6, 2, 0.7, 0, 0.5, 10.0, 5.0),
        ]
        return self.spark.createDataFrame(rows, schema=schema)

    def test_output_order_and_ratios(self):
        result = build_full_tabular(self.make_frame(), ("shared",))
        self.assertEqual(
            tuple(result.columns),
            AUDIT_COLUMNS + ("shared",) + GLOBAL_SCALAR_COLUMNS + DERIVED_SCALAR_COLUMNS,
        )
        rows = {row[TRACK_ID]: row for row in result.collect()}
        self.assertAlmostEqual(rows["valid"]["fade_in_ratio"], 0.01)
        self.assertAlmostEqual(rows["valid"]["fade_out_ratio"], 0.10)
        self.assertAlmostEqual(rows["valid"]["active_audio_ratio"], 0.89)
        self.assertAlmostEqual(rows["tolerated"]["fade_out_ratio"], 0.0)
        self.assertAlmostEqual(rows["tolerated"]["active_audio_ratio"], 0.99)

    def test_invalid_values_become_null(self):
        row = build_full_tabular(self.make_frame(), ("shared",)).where(
            f"{TRACK_ID} = 'invalid'"
        ).first()
        self.assertIsNone(row["tempo"])
        self.assertIsNone(row["key"])
        self.assertIsNone(row["mode"])
        self.assertIsNone(row["time_signature"])
        for column in DERIVED_SCALAR_COLUMNS:
            self.assertIsNone(row[column])


if __name__ == "__main__":
    unittest.main()
