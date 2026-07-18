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




if __name__ == "__main__":
    unittest.main()

