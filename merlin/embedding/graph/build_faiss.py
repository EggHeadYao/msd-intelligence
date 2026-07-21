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

from merlin.embedding.graph.config import WORD2VEC_VECTOR_SIZE


INDEX_NAME = "index_graph.faiss"
MAPPING_NAME = "index_graph_track_ids.parquet"
METADATA_NAME = "graph_faiss_metadata.json"
ENCODER_METADATA_NAME = "graph_encoder_metadata.json"
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
        return fallback
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    dimension = int(metadata["output"]["dimension"])
    require(
        dimension == fallback, "requested dimension conflicts with encoder metadata"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for FAISS index and mapping files",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf"],
        help="FAISS index type (flat for exact, ivf for faster approximate)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Read embeddings using PyArrow
    print(f"Reading embeddings from {args.embeddings}")
    table = pq.read_table(args.embeddings)
    
    track_ids = table["track_id"].to_pylist()
    embeddings_list = table["embedding"].to_pylist()
    
    print(f"Processing {len(track_ids)} songs...")

    # Convert embeddings to numpy array
    vectors = []
    for emb in embeddings_list:
        if isinstance(emb, list):
            vectors.append(emb)
        else:
            vectors.append(list(emb))

    vectors_np = np.array(vectors, dtype=np.float32)
    dim = vectors_np.shape[1]

    print(f"Building FAISS index: {vectors_np.shape[0]} vectors, dim={dim}")

    # Build FAISS index
    if args.index_type == "flat":
        index = faiss.IndexFlatL2(dim)  # L2 distance
        index.add(vectors_np)
    else:  # ivf
        n_list = max(100, len(track_ids) // 100)
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_list)
        index.train(vectors_np)
        index.add(vectors_np)

    # Write outputs
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    index_path = str(output_path / "index_graph.faiss")
    faiss.write_index(index, index_path)
    print(f"[OK] Saved FAISS index to {index_path}")

    # Write track_id mapping
    mapping = {i: track_id for i, track_id in enumerate(track_ids)}
    mapping_path = output_path / "graph_embeddings_id_map.json"
    with mapping_path.open("w") as f:
        json.dump(mapping, f)
    print(f"[OK] Saved ID mapping to {mapping_path}")

    # Write metadata
    metadata = {
        "index_type": args.index_type,
        "total_songs": len(track_ids),
        "embedding_dimension": dim,
        "index_file": "index_graph.faiss",
        "mapping_file": "graph_embeddings_id_map.json",
    }
    metadata_path = output_path / "faiss_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved metadata to {metadata_path}")

    print(f"\n[OK] FAISS index built: {len(track_ids)} songs indexed")


if __name__ == "__main__":
    main()

