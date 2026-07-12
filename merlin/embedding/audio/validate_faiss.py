from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


TRACK_ID_COLUMN = "track_id"
EMBEDDING_COLUMN = "embedding"
DEFAULT_AUDIO_DIR = Path("parquets/merlin/audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the MERLIN C1 audio FAISS index.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_AUDIO_DIR / "song_embeddings_audio.parquet",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--index-name", default="index_audio.faiss")
    parser.add_argument("--track-ids-name", default="index_audio_track_ids.parquet")
    parser.add_argument("--expected-rows", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinValidateAudioFaiss")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_selected_k(output_dir: Path) -> int | None:
    metadata_path = output_dir / "audio_encoder_metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return int(metadata["selected_k"])


def validate_mapping(mapping: DataFrame, expected_rows: int) -> None:
    stats = mapping.agg(
        F.count("*").alias("rows"),
        F.countDistinct("row_id").alias("distinct_row_id"),
        F.countDistinct(TRACK_ID_COLUMN).alias("distinct_track_id"),
        F.min("row_id").alias("min_row_id"),
        F.max("row_id").alias("max_row_id"),
        F.sum(
            F.when(
                F.col("row_id").isNull() | F.col(TRACK_ID_COLUMN).isNull(),
                1,
            ).otherwise(0),
        ).alias("null_rows"),
    ).first()
    require(stats["rows"] == expected_rows, "track-id mapping row count mismatch")
    require(stats["distinct_row_id"] == expected_rows, "track-id mapping row_id is not unique")
    require(stats["distinct_track_id"] == expected_rows, "track-id mapping track_id is not unique")
    require(stats["min_row_id"] == 0, "track-id mapping must start at row_id 0")
    require(stats["max_row_id"] == expected_rows - 1, "track-id mapping max row_id mismatch")
    require(stats["null_rows"] == 0, "track-id mapping contains null rows")


