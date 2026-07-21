from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pyspark.ml.feature import PCAModel, StandardScalerModel, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from columns import PREPARED_AUDIO_COLUMNS, TRACK_ID_COLUMN
from frozen_preprocess import apply_frozen_preprocess
from l1_stats import bootstrap_hedges_g_ci, distribution, hedges_g, preservation_summary
from lineage import sha256_path
from preprocess import add_scalar_availability
from train_pca import (
    EMBEDDING_COLUMN,
    FEATURES_COLUMN,
    PCA_FEATURES_COLUMN,
    SCALED_FEATURES_COLUMN,
    add_normalized_embedding,
)
from validate import read_metadata, require, validate_layout, validate_metadata


VALIDATION_VERSION = "c1_l1_1_v1"
PAIR_TYPES = ("same_artist", "same_release", "random")
METADATA_COLUMNS = (
    TRACK_ID_COLUMN,
    "song_id",
    "artist_id",
    "release_7digitalid",
    "year",
    "has_year",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MERLIN C1 L1-1 feature sanity validation.")
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=Path("parquets_new/prepared/song_audio_features_raw.parquet"),
    )
    parser.add_argument(
        "--songs-metadata",
        type=Path,
        default=Path("parquets_new/prepared/songs_metadata.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("parquets_new/merlin/audio"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--pair-count", type=int, default=10_000)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--reproduction-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--allow-partial-pairs",
        action="store_true",
        help="Smoke-only mode: report fewer than the required pairs instead of failing.",
    )
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinValidateC1FeatureSanity")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def require_columns(df: DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    require(not missing, f"{name} is missing columns: {missing}")


def array_dot(left: F.Column, right: F.Column) -> F.Column:
    products = F.zip_with(left, right, lambda x, y: x.cast("double") * y.cast("double"))
    return F.aggregate(products, F.lit(0.0), lambda total, value: total + value)


def cosine(left: F.Column, right: F.Column) -> F.Column:
    denominator = F.sqrt(array_dot(left, left) * array_dot(right, right))
    return array_dot(left, right) / denominator


def add_coverage(raw: DataFrame, feature_columns: Sequence[str]) -> DataFrame:
    availability = add_scalar_availability(raw)
    coverage_columns = tuple(
        column for column in feature_columns
        if column.startswith("has_") and column in availability.columns
    )
    require(bool(coverage_columns), "C1 feature coverage columns are unavailable")
    score = F.lit(0)
    for column in coverage_columns:
        score = score + F.when(F.col(column).cast("double") == 1.0, F.lit(1)).otherwise(F.lit(0))
    return availability.select(TRACK_ID_COLUMN, score.cast("int").alias("feature_coverage"))

