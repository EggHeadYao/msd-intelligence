from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pyspark import cloudpickle
from pyspark.sql import DataFrame, SparkSession, functions as F

from .features import Point

cloudpickle.register_pickle_by_value(sys.modules[__name__])

REQUIRED_COLUMNS = (
    "track_id", "artist_id", "year", "normalized_year", "features", "split",
)


def parquet_inputs(path: Path) -> list[str]:
    if path.is_dir():
        files = sorted(item for item in path.rglob("*.parquet") if not item.name.startswith((".", "_")))
        if not files:
            raise FileNotFoundError(f"no Parquet files found under {path}")
        return [str(item) for item in files]
    return [str(path)]


def load_frame(spark: SparkSession, input_path: Path, splits: tuple[str, ...]) -> DataFrame:
    source = spark.read.option("basePath", str(input_path.resolve())).parquet(*parquet_inputs(input_path))
    missing = sorted(set(REQUIRED_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    selected = source.where(F.col("split").isin(*splits)).select(*REQUIRED_COLUMNS)
    invalid = F.exists(
        "features",
        lambda value: value.isNull() | F.isnan(value) | (F.abs(value) == F.lit(float("inf"))),
    )
    summary = selected.agg(
        F.count("*").alias("count"), F.min(F.size("features")).alias("minimum_dimension"),
        F.max(F.size("features")).alias("maximum_dimension"),
        F.sum(F.when(F.col("features").isNull() | invalid, 1).otherwise(0)).alias("invalid"),
    ).first()
    if not summary or int(summary["count"]) <= 0 or int(summary["invalid"]) != 0:
        raise ValueError("input rows are empty or invalid")
    if int(summary["minimum_dimension"]) != int(summary["maximum_dimension"]):
        raise ValueError("input feature dimensions differ")
    return selected


def row_point(row) -> Point:
    label = float(row["normalized_year"])
    values = np.asarray(row["features"], dtype=np.float64)
    if not math.isfinite(label):
        raise ValueError("normalized target is not finite")
    return str(row["track_id"]), str(row["artist_id"]), int(row["year"]), label, values


def frame_points(frame: DataFrame):
    return frame.select(*REQUIRED_COLUMNS[:-1]).rdd.map(row_point)
