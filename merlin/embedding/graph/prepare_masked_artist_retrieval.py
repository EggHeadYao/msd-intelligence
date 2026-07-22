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


def create_spark(shuffle_partitions: int) -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    return (
        SparkSession.builder.appName("MerlinC2MaskedArtistPrepare")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.pyspark.python", sys.executable)
        .getOrCreate()
    )


def allocate_balanced_quotas(
    available: dict[str, int],
    requested: int,
) -> dict[str, int]:
    """Allocate near-equal deterministic quotas, redistributing short strata."""
    if requested <= 0:
        raise ValueError("requested query count must be positive")
    if set(available) != set(STRATA):
        raise ValueError(f"availability must contain exactly {STRATA}")
    if any(value < 0 for value in available.values()):
        raise ValueError("stratum availability cannot be negative")
    if sum(available.values()) < requested:
        raise ValueError("not enough eligible artists for the requested queries")

    quotas = {name: 0 for name in STRATA}
    remaining = requested
    while remaining:
        progressed = False
        for name in STRATA:
            if quotas[name] >= available[name]:
                continue
            quotas[name] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("query quota allocation made no progress")
    return quotas


def _require_columns(frame: DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = set(required) - set(frame.columns)
    require(not missing, f"{name} is missing columns: {sorted(missing)}")


def _valid_string(column: str) -> F.Column:
    return F.col(column).isNotNull() & (F.length(F.trim(F.col(column))) > 0)
