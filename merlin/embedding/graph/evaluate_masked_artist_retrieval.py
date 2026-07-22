"""Evaluate C2 retrieval after masking each query's artist relation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from merlin.embedding.graph.build_faiss import INDEX_NAME, MAPPING_NAME
from merlin.embedding.graph.prepare_masked_artist_retrieval import EXPERIMENT_VERSION
from merlin.embedding.graph.retrieval_metrics import (
    DEFAULT_CUTOFFS,
    macro_average,
    random_expectation,
    score_ranking,
)


REPORT_VERSION = "c2_l1_2_report_v1"
REPORT_NAME = "report.json"
QUERY_METRICS_NAME = "query_metrics.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument("--overfetch", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def valid_song_key(track_id: str, song_id: str | None) -> str:
    if song_id is not None and song_id.strip():
        return f"song:{song_id}"
    return f"track:{track_id}"


def stable_rank_key(seed: int, query_id: str, candidate_id: str) -> bytes:
    value = f"{seed}\x00{query_id}\x00{candidate_id}".encode("ascii")
    return hashlib.sha256(value).digest()
