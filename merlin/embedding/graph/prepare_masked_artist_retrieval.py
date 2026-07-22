"""Prepare graph inputs and labels for masked-artist C2 retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from functools import reduce
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


EXPERIMENT_VERSION = "c2_l1_2_masked_relation_v1"
STRATA = ("2", "3_5", "6_20", "21_plus")
QUERY_COLUMNS = (
    "query_track_id",
    "artist_id",
    "song_id",
    "release_7digitalid",
    "song_hotttnesss",
    "artist_track_count",
    "positive_count",
    "release_degree",
    "candidate_catalog_size",
    "connectable",
    "stratum",
)
GRAPH_COLUMNS = (
    "src_id",
    "dst_id",
    "src_type",
    "dst_type",
    "edge_type",
    "directed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--output-partitions", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()
