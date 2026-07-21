"""Validate the persisted MERLIN C2 graph embedding and FAISS artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from merlin.embedding.graph.build_faiss import (
    ENCODER_METADATA_NAME,
    INDEX_NAME,
    MAPPING_NAME,
    METADATA_NAME,
    NORM_TOLERANCE,
)
from merlin.embedding.graph.config import WORD2VEC_VECTOR_SIZE
from merlin.embedding.graph.train_word2vec import EMBEDDINGS_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1_000_000)
    parser.add_argument("--dimension", type=int, default=WORD2VEC_VECTOR_SIZE)
    parser.add_argument("--queries", type=int, default=20)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing metadata file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
