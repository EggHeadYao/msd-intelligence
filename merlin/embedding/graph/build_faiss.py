"""MERLIN C2: Build FAISS index for graph embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to song_embeddings_graph.parquet",
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
    print(f"✓ Saved FAISS index to {index_path}")

    # Write track_id mapping
    mapping = {i: track_id for i, track_id in enumerate(track_ids)}
    mapping_path = output_path / "graph_embeddings_id_map.json"
    with mapping_path.open("w") as f:
        json.dump(mapping, f)
    print(f"✓ Saved ID mapping to {mapping_path}")

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
    print(f"✓ Saved metadata to {metadata_path}")

    print(f"\n✓ FAISS index built: {len(track_ids)} songs indexed")


if __name__ == "__main__":
    main()

