"""C2 meta-path guided random walk generator (core logic)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from merlin.embedding.graph.config import EDGE_SCHEMA, META_PATHS, META_PATH_WEIGHTS

# Maps adjacency Parquet file basename (without .parquet) to edge_type key.
_FILE_EDGE_MAP: dict[str, str] = {
    "fwd_song_artist": "song_artist",
    "fwd_song_album": "song_album",
    "fwd_song_tag": "song_tag",
    "fwd_song_similar_artist": "song_similar_artist",
    "fwd_song_year": "song_year",
    "rev_song_artist": "song_artist",
    "rev_song_album": "song_album",
    "rev_song_tag": "song_tag",
    "rev_song_year": "song_year",
    "rev_artist_tag_fwd": "artist_tag",
    "rev_artist_tag_rev": "artist_tag",
    "rev_artist_similarity": "artist_similarity",
}


def load_adjacency(
    adj_dir: str,
    node_to_int: dict[str, int],
) -> tuple[
    dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
]:
    """Load adjacency Parquet files into two dicts.

    Returns:
        song_adj: {song_int_id: {edge_type: (neighbor_ints, weights)}}
        node_adj: {non_song_int_id: {edge_type: (neighbor_ints, weights)}}
    """
    import pyarrow.parquet as pq

    song_adj: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    node_adj: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for fname in sorted(os.listdir(adj_dir)):
        if not fname.endswith(".parquet"):
            continue
        base: str = Path(fname).stem
        if base not in _FILE_EDGE_MAP:
            continue

        edge_type: str = _FILE_EDGE_MAP[base]
        is_fwd: bool = base.startswith("fwd_")

        table = pq.read_table(os.path.join(adj_dir, fname))
        for row in table.to_pylist():
            node_str: str = row["node_str"]
            if node_str not in node_to_int:
                continue
            node_int: int = node_to_int[node_str]

            nids: np.ndarray = np.frombuffer(
                row["neighbor_ids"],
                dtype=np.int32,
            )
            wts: np.ndarray = np.frombuffer(
                row["weights"],
                dtype=np.float32,
            )

            target = song_adj if is_fwd else node_adj
            target.setdefault(node_int, {})[edge_type] = (nids, wts)

    return song_adj, node_adj
