from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "data"
sys.path.insert(0, str(DATA_DIR))

from build_dataset import assign_artists  # noqa: E402
from columns import ARTIST_ID, SPLIT, TEST, TRACK_ID, YEAR  # noqa: E402


class DatasetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("YearPredictionDatasetContractTest")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_official_test_assignment_overrides_hash(self):
        artists = self.spark.createDataFrame([("artist-a",), ("artist-test",)], [ARTIST_ID])
        official_test = self.spark.createDataFrame([("artist-test",)], [ARTIST_ID])
        result = {
            row[ARTIST_ID]: row[SPLIT]
            for row in assign_artists(artists, official_test, 472, 10).collect()
        }
        self.assertEqual(result["artist-test"], TEST)

    def test_assignment_is_deterministic_and_artist_level(self):
        tracks = self.spark.createDataFrame(
            [
                ("track-1", "artist-a", 1990),
                ("track-2", "artist-a", 1991),
                ("track-3", "artist-b", 2000),
            ],
            [TRACK_ID, ARTIST_ID, YEAR],
        )
        artists = tracks.select(ARTIST_ID).distinct()
        official_test = self.spark.createDataFrame([], f"{ARTIST_ID} string")
        first = assign_artists(artists, official_test, 472, 10)
        second = assign_artists(artists, official_test, 472, 10)
        self.assertEqual(
            sorted(tuple(row) for row in first.collect()),
            sorted(tuple(row) for row in second.collect()),
        )
        labeled = tracks.join(first, ARTIST_ID)
        artist_a_splits = labeled.where(F.col(ARTIST_ID) == "artist-a").select(SPLIT).distinct()
        self.assertEqual(artist_a_splits.count(), 1)

    def test_null_year_is_not_a_label(self):
        scalar = self.spark.createDataFrame(
            [("track-labeled", "artist-a", 1990), ("track-unlabeled", "artist-b", None)],
            f"{TRACK_ID} string, {ARTIST_ID} string, {YEAR} int",
        )
        labeled = scalar.where(F.col(YEAR).isNotNull())
        self.assertEqual([row[TRACK_ID] for row in labeled.collect()], ["track-labeled"])


if __name__ == "__main__":
    unittest.main()
