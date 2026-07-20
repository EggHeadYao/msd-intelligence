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


def load_training_data(
    spark: SparkSession,
    input_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> TrainingData:
    frame = spark.read.parquet(spark_path(input_path))
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected linear feature columns: {frame.columns}")
    frame = frame.where(F.col(SPLIT).isin(TRAIN, VALIDATION))
    invalid_element = F.exists(
        F.col(FEATURES),
        lambda value: value.isNull()
        | F.isnan(value)
        | (F.abs(value) == F.lit(float("inf"))),
    )
    invalid_label = (
        F.col(LABEL).isNull()
        | F.isnan(LABEL)
        | (F.abs(F.col(LABEL)) == F.lit(float("inf")))
        | (F.col(LABEL) < F.lit(0.0))
        | (F.col(LABEL) > F.lit(1.0))
    )
    rows = frame.groupBy(SPLIT).agg(
        F.count(F.lit(1)).alias("count"),
        F.countDistinct(TRACK_ID).alias("distinct_tracks"),
        F.min(F.size(FEATURES)).alias("minimum_dimension"),
        F.max(F.size(FEATURES)).alias("maximum_dimension"),
        F.avg(LABEL).alias("label_mean"),
        F.sum(
            F.when(F.col(FEATURES).isNull() | invalid_element | invalid_label, 1).otherwise(0)
        ).alias("invalid_rows"),
    ).collect()
    summaries = {row[SPLIT]: row.asDict() for row in rows}
    if set(summaries) != {TRAIN, VALIDATION}:
        raise ValueError("training data must contain train and validation splits")
    dimensions: set[int] = set()
    counts: dict[str, int] = {}
    label_means: dict[str, float] = {}
    for split, summary in summaries.items():
        count = int(summary["count"])
        counts[split] = count
        label_means[split] = float(summary["label_mean"])
        if count <= 0 or int(summary["distinct_tracks"]) != count:
            raise ValueError(f"{split} track IDs are empty or duplicated")
        if int(summary["invalid_rows"]) != 0:
            raise ValueError(f"{split} contains invalid labels or feature vectors")
        minimum = int(summary["minimum_dimension"])
        maximum = int(summary["maximum_dimension"])
        if minimum <= 0 or minimum != maximum:
            raise ValueError(f"{split} feature dimensions are empty or inconsistent")
        dimensions.add(minimum)
    if len(dimensions) != 1:
        raise ValueError("train and validation feature dimensions differ")
    dimension = dimensions.pop()
    total_count = sum(counts.values())
    if frame.select(TRACK_ID).distinct().count() != total_count:
        raise ValueError("track IDs overlap across train and validation")
    artist_overlap = (
        frame.groupBy(ARTIST_ID)
        .agg(F.countDistinct(SPLIT).alias("split_count"))
        .where(F.col("split_count") > 1)
        .limit(1)
        .count()
    )
    if artist_overlap:
        raise ValueError("artists overlap across train and validation")
    if metadata is not None:
        if dimension != expected_dimension(metadata):
            raise ValueError("feature dimension differs from preprocessing metadata")
        expected_counts = {
            split: int(metadata["counts"]["splits"][split])
            for split in (TRAIN, VALIDATION)
        }
        if counts != expected_counts:
            raise ValueError("train or validation count differs from preprocessing metadata")
    if not math.isfinite(float(dimension)):
        raise ValueError("feature dimension is not finite")
    return TrainingData(
        frame=frame,
        dimension=dimension,
        counts=counts,
        label_means=label_means,
    )
