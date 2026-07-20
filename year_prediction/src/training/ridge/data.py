from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark import RDD
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from objectives import Point
from target import target_contract


TRACK_ID = "track_id"
ARTIST_ID = "artist_id"
YEAR = "year"
LABEL = "normalized_year"
FEATURES = "features"
SPLIT = "split"
TRAIN = "train"
VALIDATION = "validation"
CONTRACT_VERSION = "year_prediction_t90_training_v1"
EXPECTED_DIMENSION = 90
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


def read_training_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="ascii") as handle:
        manifest: dict[str, Any] = json.load(handle)
    validate_training_manifest(manifest)
    return manifest




def expected_dimension(manifest: dict[str, Any]) -> int:
    return int(manifest["preprocessing"]["dimension"])


def expected_split_counts(manifest: dict[str, Any]) -> dict[str, int]:
    return {
        split: int(manifest["counts"]["splits"][split]["tracks"])
        for split in (TRAIN, VALIDATION)
    }


def load_training_data(
    spark: SparkSession,
    input_path: str | Path,
    manifest: dict[str, Any] | None = None,
) -> TrainingData:
    frame = spark.read.parquet(spark_path(input_path))
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected T90 vector columns: {frame.columns}")
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
        | ~F.col(LABEL).between(0.0, 1.0)
    )
    rows = frame.groupBy(SPLIT).agg(
        F.count("*").alias("count"),
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
        raise ValueError("Training data must contain train and validation splits")
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
        raise ValueError("Train and validation feature dimensions differ")
    dimension = dimensions.pop()
    if frame.select(TRACK_ID).distinct().count() != sum(counts.values()):
        raise ValueError("Track IDs overlap across train and validation")
    artist_overlap = (
        frame.groupBy(ARTIST_ID)
        .agg(F.countDistinct(SPLIT).alias("split_count"))
        .where(F.col("split_count") > 1)
        .limit(1)
        .count()
    )
    if artist_overlap:
        raise ValueError("Artists overlap across train and validation")
    if manifest is not None:
        validate_training_manifest(manifest)
        if dimension != expected_dimension(manifest):
            raise ValueError("Feature dimension differs from T90 manifest")
        if counts != expected_split_counts(manifest):
            raise ValueError("Train or validation count differs from T90 manifest")
    return TrainingData(
        frame=frame,
        dimension=dimension,
        counts=counts,
        label_means=label_means,
    )
