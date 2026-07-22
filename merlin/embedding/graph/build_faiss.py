"""Build the exact inner-product FAISS index for MERLIN C2 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from merlin.embedding.graph.config import (
    GRAPH_CONTRACT_KEY,
    GRAPH_CONTRACT_VERSION,
    SEED,
    WORD2VEC_MAX_ITER,
    WORD2VEC_MAX_SENTENCE_LENGTH,
    WORD2VEC_MIN_COUNT,
    WORD2VEC_NUM_PARTITIONS,
    WORD2VEC_STEP_SIZE,
    WORD2VEC_VECTOR_SIZE,
    WORD2VEC_WINDOW_SIZE,
)


INDEX_NAME = "index_graph.faiss"
MAPPING_NAME = "index_graph_track_ids.parquet"
METADATA_NAME = "index_graph_manifest.json"
ENCODER_METADATA_NAME = "graph_encoder_metadata.json"
FAISS_MANIFEST_VERSION = "merlin_faiss_index_v1"
REQUIRED_COLUMNS = ("node_id", "track_id", "embedding")
NORM_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1_000_000)
    parser.add_argument("--dimension", type=int, default=WORD2VEC_VECTOR_SIZE)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prepare_output(output: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / INDEX_NAME
    mapping_path = output / MAPPING_NAME
    metadata_path = output / METADATA_NAME
    existing = [
        path for path in (index_path, mapping_path, metadata_path) if path.exists()
    ]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"C2 FAISS output already exists: {names}")
    for path in existing:
        remove_path(path)
    return index_path, mapping_path, metadata_path


def read_expected_dimension(output: Path, fallback: int) -> int:
    metadata_path = output / ENCODER_METADATA_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing graph encoder metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    require(
        metadata.get(GRAPH_CONTRACT_KEY) == GRAPH_CONTRACT_VERSION,
        "graph encoder contract version mismatch",
    )
    require(
        metadata.get("embedding_source") == "direct_word2vec_track_token",
        "graph embedding source mismatch",
    )
    dimension = int(metadata["output"]["dimension"])
    require(
        dimension == fallback, "requested dimension conflicts with encoder metadata"
    )
    training = metadata.get("training", {})
    expected_training = {
        "vector_size": WORD2VEC_VECTOR_SIZE,
        "window_size": WORD2VEC_WINDOW_SIZE,
        "min_count": WORD2VEC_MIN_COUNT,
        "max_iter": WORD2VEC_MAX_ITER,
        "step_size": WORD2VEC_STEP_SIZE,
        "num_partitions": WORD2VEC_NUM_PARTITIONS,
        "max_sentence_length": WORD2VEC_MAX_SENTENCE_LENGTH,
        "seed": SEED,
    }
    require(
        training == expected_training,
        "graph encoder training parameters do not match the frozen C2 contract",
    )
    return dimension


def read_embeddings(path: Path, expected_rows: int) -> pa.Table:
    table = pq.read_table(path, columns=list(REQUIRED_COLUMNS))
    require(table.num_rows == expected_rows, "embedding row count mismatch")
    require(table.column_names == list(REQUIRED_COLUMNS), "embedding columns mismatch")
    require(table["node_id"].null_count == 0, "embedding node_id contains null")
    require(table["track_id"].null_count == 0, "embedding track_id contains null")
    require(table["embedding"].null_count == 0, "embedding vector contains null")
    require(
        int(pc.count_distinct(table["node_id"]).as_py()) == expected_rows,
        "embedding node_id is not unique",
    )
    require(
        int(pc.count_distinct(table["track_id"]).as_py()) == expected_rows,
        "embedding track_id is not unique",
    )
    return table.sort_by([("node_id", "ascending")])


def embedding_matrix(array: pa.Array, dimension: int) -> np.ndarray:
    require(array.null_count == 0, "embedding batch contains null vectors")
    lengths = pc.list_value_length(array).to_numpy(zero_copy_only=False)
    require(
        np.all(lengths == dimension),
        "embedding vector dimension mismatch",
    )
    flattened = pc.list_flatten(array).to_numpy(zero_copy_only=False)
    matrix = np.asarray(flattened, dtype=np.float32).reshape(len(array), dimension)
    require(np.all(np.isfinite(matrix)), "embedding contains NaN or infinite values")
    norms = np.linalg.norm(matrix, axis=1)
    require(
        np.all(np.abs(norms - 1.0) <= NORM_TOLERANCE),
        "embedding is not L2 normalized",
    )
    return np.ascontiguousarray(matrix, dtype=np.float32)


def build_index(
    table: pa.Table,
    dimension: int,
    batch_size: int,
) -> faiss.IndexFlatIP:
    require(batch_size > 0, "batch size must be positive")
    index = faiss.IndexFlatIP(dimension)
    embedding_index = table.schema.get_field_index("embedding")
    previous_node_id: int | None = None

    for batch in table.to_batches(max_chunksize=batch_size):
        node_ids = batch.column("node_id").to_numpy(zero_copy_only=False)
        if previous_node_id is not None:
            require(
                int(node_ids[0]) > previous_node_id, "node IDs are not strictly ordered"
            )
        require(
            node_ids.size == 1 or np.all(node_ids[1:] > node_ids[:-1]),
            "node IDs are not strictly ordered",
        )
        previous_node_id = int(node_ids[-1])
        matrix = embedding_matrix(batch.column(embedding_index), dimension)
        index.add(matrix)

    require(index.ntotal == table.num_rows, "FAISS row count mismatch")
    return index


def write_mapping(table: pa.Table, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    remove_path(temporary)
    temporary.mkdir(parents=True)
    mapping = pa.table(
        {
            "row_id": pa.array(np.arange(table.num_rows, dtype=np.int64)),
            "node_id": table["node_id"].combine_chunks(),
            "track_id": table["track_id"].combine_chunks(),
        },
    )
    pq.write_table(
        mapping,
        temporary / "part-00000.parquet",
        compression="snappy",
    )
    (temporary / "_SUCCESS").touch()
    temporary.replace(path)


def validate_index(
    index: faiss.IndexFlatIP, rows: int, dimension: int
) -> dict[str, Any]:
    require(index.ntotal == rows, "FAISS index size mismatch")
    require(index.d == dimension, "FAISS index dimension mismatch")
    require(index.metric_type == faiss.METRIC_INNER_PRODUCT, "FAISS metric mismatch")

    sample_ids = sorted({0, rows // 4, rows // 2, (3 * rows) // 4, rows - 1})
    queries = np.vstack([index.reconstruct(row_id) for row_id in sample_ids]).astype(
        np.float32,
        copy=False,
    )
    scores, neighbors = index.search(queries, min(10, rows))
    for position, row_id in enumerate(sample_ids):
        require(row_id in neighbors[position], "FAISS sample did not retrieve itself")
        require(scores[position][0] <= 1.0001, "FAISS inner-product score exceeds one")
        require(scores[position][0] >= 0.9999, "FAISS self score is below one")
    return {
        "sample_row_ids": sample_ids,
        "sample_top_scores": [
            float(scores[index_value][0]) for index_value in range(len(sample_ids))
        ],
    }


def sha256_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    digest = hashlib.sha256()
    files = (path,) if path.is_file() else tuple(
        sorted(item for item in path.rglob("*") if item.is_file())
    )
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    require(args.expected_rows > 0, "expected row count must be positive")
    require(args.dimension > 0, "embedding dimension must be positive")
    index_path, mapping_path, metadata_path = prepare_output(
        args.output, args.overwrite
    )
    dimension = read_expected_dimension(args.output, args.dimension)

    print(f"graph_faiss_read_start embeddings={args.embeddings}")
    table = read_embeddings(args.embeddings, args.expected_rows)
    print(f"graph_faiss_build_start rows={table.num_rows}, dimension={dimension}")
    index = build_index(table, dimension, args.batch_size)
    validation = validate_index(index, table.num_rows, dimension)

    temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
    remove_path(temporary_index)
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(index_path)
    write_mapping(table, mapping_path)

    encoder_metadata_path = args.output / ENCODER_METADATA_NAME
    metadata = {
        "artifact_type": "merlin_faiss_index",
        "manifest_version": FAISS_MANIFEST_VERSION,
        "embedding_space": "graph",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        GRAPH_CONTRACT_KEY: GRAPH_CONTRACT_VERSION,
        "input_embeddings": str(args.embeddings.resolve()),
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "dimension": int(index.d),
        "row_count": int(index.ntotal),
        "index_file": INDEX_NAME,
        "mapping_path": MAPPING_NAME,
        "index_sha256": sha256_path(index_path),
        "mapping_sha256": sha256_path(mapping_path),
        "encoder_metadata_sha256": sha256_path(encoder_metadata_path),
        "mapping_columns": ["row_id", "node_id", "track_id"],
        "mapping_order": "node_id_ascending",
        "validation": validation,
    }
    write_metadata(metadata_path, metadata)
    print(
        "graph_faiss_build_done "
        f"rows={index.ntotal}, dimension={index.d}, index={index_path}, "
        f"mapping={mapping_path}",
    )


if __name__ == "__main__":
    main()
