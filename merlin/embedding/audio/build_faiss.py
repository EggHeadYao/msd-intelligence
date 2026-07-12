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

