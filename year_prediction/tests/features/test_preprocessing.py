from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

FEATURES_DIR = Path(__file__).resolve().parents[2] / "src" / "features"
sys.path.insert(0, str(FEATURES_DIR))

from columns import SEGMENT_COLUMNS  # noqa: E402
from preprocessing import fit_feature_contract, transform_features, validate_binary_columns  # noqa: E402


def row(index: int, split: str, **overrides):
    value = {
        "track_id": f"TR{index}",
        "artist_id": f"AR{index}",
        "year": 1990 + index,
        "danceability": 0.1 + index * 0.1,
        "energy": 0.2 + index * 0.1,
        "loudness": -20.0 + index,
        "tempo": 90.0 + index * 10.0,
        "duration": 180.0 + index * 10.0,
        "key": index % 12,
        "mode": index % 2,
        "time_signature": 3 + index % 2,
        "has_segments": 1,
        "split": split,
    }
    value.update({column: float(index + offset / 100.0) for offset, column in enumerate(SEGMENT_COLUMNS)})
    value.update(overrides)
    return value


class PreprocessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("YearFeaturePreprocessingTest")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_fit_uses_training_rows_only(self):
        rows = [row(index, "train") for index in range(4)]
        rows.append(row(9, "validation", tempo=10000.0, time_signature=7))
        data = self.spark.createDataFrame(rows)
        state = fit_feature_contract(data.where(F.col("split") == "train"), quantile_error=0.0)

        self.assertNotIn(7, state["time_signature_values"])
        self.assertLessEqual(state["clip_bounds"]["tempo"][1], 120.0)
        transformed = transform_features(data.where(F.col("split") == "validation"), state).first()
        self.assertEqual(transformed["time_signature"], -1)
        self.assertEqual(transformed["time_signature_unknown"], 1.0)
        self.assertTrue(math.isfinite(transformed["tempo_log"]))

    def test_missing_segments_use_training_means(self):
        rows = [row(index, "train") for index in range(3)]
        missing = row(4, "validation", has_segments=0)
        missing.update({column: 0.0 for column in SEGMENT_COLUMNS})
        data = self.spark.createDataFrame([*rows, missing])
        train = data.where(F.col("split") == "train")
        state = fit_feature_contract(train, quantile_error=0.0)
        transformed = transform_features(data.where(F.col("split") == "validation"), state).first()

        self.assertEqual(transformed["has_segments"], 0.0)
        for column in SEGMENT_COLUMNS:
            self.assertAlmostEqual(transformed[column], state["segment_means"][column])

    def test_invalid_binary_values_are_rejected(self):
        data = self.spark.createDataFrame([row(0, "train", mode=3)])
        with self.assertRaises(ValueError):
            validate_binary_columns(data)


if __name__ == "__main__":
    unittest.main()

