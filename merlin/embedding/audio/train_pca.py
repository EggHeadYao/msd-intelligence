from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vector
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from columns import TRACK_ID_COLUMN
from preprocess import preprocess_audio_features


FEATURES_COLUMN = "features"
SCALED_FEATURES_COLUMN = "scaled_features"
PCA_FEATURES_COLUMN = "pca_features"
EMBEDDING_COLUMN = "embedding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MERLIN C1 PCA audio encoder.")
    parser.add_argument("--input", type=Path, default=Path("parquets/prepared/song_audio_features_raw.parquet"))
    parser.add_argument("--output", type=Path, default=Path("parquets/merlin/audio"))
    parser.add_argument("--target-variance", type=float, default=0.95)
    parser.add_argument("--fixed-k", type=int, default=0)
    parser.add_argument("--max-components", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinTrainAudioPCA")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def cumulative(values: list[float]) -> list[float]:
    total = 0.0
    result = []
    for value in values:
        total += float(value)
        result.append(total)
    return result

