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


def validate_faiss(
    output: Path,
    expected_rows: int,
    dimension: int,
    query_count: int,
) -> None:
    index_path = output / INDEX_NAME
    require(index_path.is_file(), f"missing FAISS index: {index_path}")
    index = faiss.read_index(str(index_path))
    require(index.ntotal == expected_rows, "FAISS index row count mismatch")
    require(index.d == dimension, "FAISS index dimension mismatch")
    require(
        index.metric_type == faiss.METRIC_INNER_PRODUCT,
        "FAISS metric is not inner product",
    )

    query_count = min(query_count, expected_rows)
    sample_ids = np.linspace(0, expected_rows - 1, query_count, dtype=np.int64)
    queries = np.vstack(
        [index.reconstruct(int(row_id)) for row_id in sample_ids]
    ).astype(
        np.float32,
        copy=False,
    )
    require(np.all(np.isfinite(queries)), "FAISS reconstructed vector is not finite")
    require(
        np.all(np.abs(np.linalg.norm(queries, axis=1) - 1.0) <= NORM_TOLERANCE),
        "FAISS reconstructed vector is not normalized",
    )
    scores, neighbors = index.search(queries, min(10, expected_rows))
    for position, row_id in enumerate(sample_ids):
        require(
            int(row_id) in neighbors[position], "FAISS query did not retrieve itself"
        )
        require(0.9999 <= scores[position][0] <= 1.0001, "FAISS self score mismatch")

    faiss_metadata = read_json(output / METADATA_NAME)
    require(
        faiss_metadata["index"]["type"] == "IndexFlatIP", "FAISS metadata type mismatch"
    )
    require(
        faiss_metadata["index"]["rows"] == expected_rows, "FAISS metadata rows mismatch"
    )
    require(
        faiss_metadata["index"]["dimension"] == dimension,
        "FAISS metadata dimension mismatch",
    )
    require(
        faiss_metadata["index"]["sha256"] == sha256_file(index_path),
        "FAISS index hash mismatch",
    )


def validate_encoder_metadata(output: Path, expected_rows: int, dimension: int) -> None:
    metadata = read_json(output / ENCODER_METADATA_NAME)
    require(
        metadata["embedding_source"] == "direct_word2vec_track_token",
        "graph embedding source mismatch",
    )
    require(
        metadata["output"]["rows"] == expected_rows, "encoder metadata rows mismatch"
    )
    require(
        metadata["output"]["dimension"] == dimension,
        "encoder metadata dimension mismatch",
    )
    require(metadata["output"]["dtype"] == "float32", "encoder metadata dtype mismatch")
    require(
        metadata["output"]["l2_normalized"] is True, "encoder metadata norm mismatch"
    )


def main() -> None:
    args = parse_args()
    require(args.expected_rows > 0, "expected row count must be positive")
    require(args.dimension > 0, "embedding dimension must be positive")
    require(args.queries > 0, "query count must be positive")
    validate_encoder_metadata(args.output, args.expected_rows, args.dimension)
    validate_embeddings(args.output, args.expected_rows, args.dimension)
    validate_mapping(args.output, args.expected_rows)
    validate_faiss(args.output, args.expected_rows, args.dimension, args.queries)
    print(
        "graph_artifact_validation_passed "
        f"rows={args.expected_rows}, dimension={args.dimension}, queries={args.queries}",
    )


if __name__ == "__main__":
    main()
