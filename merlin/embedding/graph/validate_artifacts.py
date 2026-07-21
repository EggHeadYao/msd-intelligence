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


def validate_embeddings(output: Path, expected_rows: int, dimension: int) -> None:
    embeddings_path = output / EMBEDDINGS_NAME
    require(embeddings_path.is_dir(), f"missing embeddings: {embeddings_path}")
    table = pq.read_table(embeddings_path)
    require(
        table.column_names == ["node_id", "track_id", "embedding"],
        "embedding columns mismatch",
    )
    require(table.num_rows == expected_rows, "embedding row count mismatch")
    require(table["node_id"].null_count == 0, "embedding node_id contains null")
    require(table["track_id"].null_count == 0, "embedding track_id contains null")
    require(table["embedding"].null_count == 0, "embedding contains null vectors")
    require(
        int(pc.count_distinct(table["node_id"]).as_py()) == expected_rows,
        "embedding node_id is not unique",
    )
    require(
        int(pc.count_distinct(table["track_id"]).as_py()) == expected_rows,
        "embedding track_id is not unique",
    )
    lengths = pc.list_value_length(table["embedding"])
    require(
        int(pc.min(lengths).as_py()) == dimension
        and int(pc.max(lengths).as_py()) == dimension,
        "embedding dimension mismatch",
    )


def validate_mapping(output: Path, expected_rows: int) -> None:
    mapping_path = output / MAPPING_NAME
    require(mapping_path.is_dir(), f"missing FAISS mapping: {mapping_path}")
    mapping = pq.read_table(mapping_path)
    require(
        mapping.column_names == ["row_id", "node_id", "track_id"],
        "FAISS mapping columns mismatch",
    )
    require(mapping.num_rows == expected_rows, "FAISS mapping row count mismatch")
    for column in mapping.column_names:
        require(
            mapping[column].null_count == 0, f"FAISS mapping {column} contains null"
        )
        require(
            int(pc.count_distinct(mapping[column]).as_py()) == expected_rows,
            f"FAISS mapping {column} is not unique",
        )
    row_ids = mapping["row_id"].to_numpy(zero_copy_only=False)
    node_ids = mapping["node_id"].to_numpy(zero_copy_only=False)
    require(
        np.array_equal(row_ids, np.arange(expected_rows)),
        "FAISS row IDs are not contiguous",
    )
    require(
        node_ids.size == 1 or np.all(node_ids[1:] > node_ids[:-1]),
        "FAISS mapping is not ordered by node_id",
    )
