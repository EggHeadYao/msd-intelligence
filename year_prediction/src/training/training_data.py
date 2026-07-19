from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark import RDD
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from objectives import Point


TRACK_ID = "track_id"
ARTIST_ID = "artist_id"
YEAR = "year"
LABEL = "normalized_year"
FEATURES = "features"
SPLIT = "split"
TRAIN = "train"
VALIDATION = "validation"
EXPECTED_COLUMNS = (TRACK_ID, ARTIST_ID, YEAR, LABEL, FEATURES, SPLIT)


@dataclass(frozen=True)
class TrainingData:
    frame: DataFrame
    dimension: int
    counts: dict[str, int]
    label_means: dict[str, float]

    def points(self, split: str) -> RDD[Point]:
        return self.frame.where(F.col(SPLIT) == split).select(FEATURES, LABEL).rdd.map(
            lambda row: (row[FEATURES], float(row[LABEL]))
        )

    def prediction_rows(self, split: str):
        return self.frame.where(F.col(SPLIT) == split).select(
            TRACK_ID, ARTIST_ID, YEAR, LABEL, FEATURES
        ).rdd.map(
            lambda row: (
                row[TRACK_ID],
                row[ARTIST_ID],
                int(row[YEAR]),
                float(row[LABEL]),
                row[FEATURES],
            )
        )


def spark_path(value: str | Path) -> str:
    text = str(value)
    return text if "://" in text else Path(text).resolve().as_uri()


def read_feature_metadata(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="ascii") as handle:
        return json.load(handle)


def expected_dimension(metadata: dict[str, Any]) -> int:
    return int(metadata["outputs"]["linear_vectors"]["dimension"])

