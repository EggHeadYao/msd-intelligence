from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = ROOT / "src" / "training"
RIDGE_DIR = TRAINING_DIR / "ridge"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(RIDGE_DIR))

import prepare_t90  # noqa: E402
import validate_t90_data  # noqa: E402
from model_io import read_json  # noqa: E402
from target import denormalize_year, normalize_year  # noqa: E402


class PrepareT90Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("PrepareT90Test")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_target_round_trip_and_bounds(self):
        self.assertEqual(normalize_year(1922), 0.0)
        self.assertEqual(normalize_year(2011), 1.0)
        self.assertAlmostEqual(denormalize_year(normalize_year(1980)), 1980.0)
        with self.assertRaises(ValueError):
            normalize_year(1921)

    def test_constant_train_feature_is_rejected(self):
        frame = self.spark.createDataFrame(
            [("train", 1.0), ("train", 1.0), ("validation", 2.0)],
            "split string, constant double",
        )
        with self.assertRaisesRegex(ValueError, "standard deviation"):
            prepare_t90.fit_feature_statistics(frame, ("constant",))

    def test_small_model_ready_artifact(self):
        columns = tuple(f"t90_{index}" for index in range(90))
        schema = ", ".join(
            (
                "track_id string",
                "artist_id string",
                "year int",
                "split string",
                *(f"{column} double" for column in columns),
            )
        )

        def feature_values(row_index: int) -> list[float]:
            return [float((row_index + 1) * (index + 1)) for index in range(90)]

        rows = []
        splits = ("train", "train", "train", "train", "validation", "test")
        for row_index, split in enumerate(splits):
            values = feature_values(row_index)
            if row_index == 0:
                values[0] = None  # type: ignore[assignment]
