from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType


TRACK_ID_COLUMN = "track_id"
EMBEDDING_COLUMN = "embedding"
DEFAULT_AUDIO_DIR = Path("parquets/merlin/audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the MERLIN C1 audio FAISS index.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_AUDIO_DIR / "song_embeddings_audio.parquet",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--index-name", default="index_audio.faiss")
    parser.add_argument("--track-ids-name", default="index_audio_track_ids.parquet")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinBuildAudioFaiss")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expected_dimension(output_dir: Path) -> int | None:
    metadata_path = output_dir / "audio_encoder_metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return int(metadata["selected_k"])


def read_embeddings(spark: SparkSession, path: Path, limit: int) -> DataFrame:
    embeddings = spark.read.parquet(spark_path(path)).select(TRACK_ID_COLUMN, EMBEDDING_COLUMN)
    if limit > 0:
        embeddings = embeddings.orderBy(TRACK_ID_COLUMN).limit(limit)
    else:
        embeddings = embeddings.orderBy(TRACK_ID_COLUMN)
    return embeddings.persist(StorageLevel.DISK_ONLY)


def write_track_id_mapping(embeddings: DataFrame, output_path: Path) -> None:
    schema = StructType(
        (
            StructField("row_id", LongType(), nullable=False),
            StructField(TRACK_ID_COLUMN, StringType(), nullable=False),
        ),
    )
    mapping = embeddings.select(TRACK_ID_COLUMN).rdd.map(lambda row: row[0]).zipWithIndex()
    mapping = mapping.map(lambda item: (int(item[1]), item[0]))
    embeddings.sparkSession.createDataFrame(mapping, schema).write.mode("overwrite").parquet(
        spark_path(output_path),
    )


