"""Generate eligibility-aware mixed walks over the canonical C2 index."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from merlin.embedding.graph.config import ADJACENCY_NAMES, META_PATHS
from merlin.embedding.graph.index import decode_typed_key, encode_typed_key

AdjacencyEntry = tuple[np.ndarray, np.ndarray]

_ADJACENCY_TYPES: dict[str, tuple[str, str]] = {
    "track_to_artist": ("track", "artist"),
    "artist_to_tracks": ("artist", "track"),
    "track_to_release": ("track", "release"),
    "release_to_tracks": ("release", "track"),
    "artist_to_terms": ("artist", "term"),
    "term_to_artists": ("term", "artist"),
    "artist_to_similar_artists": ("artist", "artist"),
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


def resolve_edge_type(edge_spec: str) -> str:
    """Strip 'rev:' prefix from an edge-type spec."""
    if edge_spec.startswith("rev:"):
        return edge_spec[4:]
    return edge_spec


def pick_neighbor(neighbor_ids: np.ndarray, rng: np.random.Generator) -> int:
    """Uniform random pick from an int32 neighbor array.  O(1)."""
    idx: int = rng.integers(0, len(neighbor_ids))
    return int(neighbor_ids[idx])


def _step_dst_is_song(edge_spec: str) -> bool:
    """True if the destination of this meta-path step is a song node."""
    is_rev: bool = edge_spec.startswith("rev:")
    base: str = edge_spec[4:] if is_rev else edge_spec
    src_type, dst_type = EDGE_SCHEMA[base]
    neighbor_type: str = src_type if is_rev else dst_type
    return neighbor_type == "song"


def follow_meta_path(
    start_song: int,
    path_template: list[str],
    song_adj: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    node_adj: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    rng: np.random.Generator,
    target_len: int = 40,
) -> list[int]:
    """Generate one meta-path guided random walk.

    Args:
        start_song: integer ID of the starting song.
        path_template: list of edge-type specs (e.g. ["song_tag", "rev:song_tag"]).
        song_adj: per-song forward adjacency.
        node_adj: per-intermediate-node reverse adjacency.
        rng: per-walk numpy random generator.
        target_len: desired number of song nodes in the output sequence.

    Returns:
        List of song int IDs representing the walk.  Length may be
        less than *target_len* if the walk encounters a dead end.
    """
    walk_seq: list[int] = [start_song]
    current_node: int = start_song
    is_song: bool = True
    path_len: int = len(path_template)

    # Pre-compute which steps yield song nodes
    song_steps: list[bool] = [_step_dst_is_song(s) for s in path_template]

    step: int = 0
    while len(walk_seq) < target_len:
        edge_spec: str = path_template[step % path_len]
        base_type: str = resolve_edge_type(edge_spec)

        adj = song_adj if is_song else node_adj
        entry = adj.get(current_node, {}).get(base_type)
        if entry is None:
            break  # dead end

        neighbor_ids, _weights = entry
        current_node = pick_neighbor(neighbor_ids, rng)
        is_song = song_steps[step % path_len]
        if is_song:
            walk_seq.append(current_node)
        step += 1

    return walk_seq


def _choose_meta_path(rng: np.random.Generator) -> tuple[str, list[str]]:
    """Weighted random selection of a meta-path template."""
    names: list[str] = list(META_PATH_WEIGHTS.keys())
    weights: np.ndarray = np.array(
        [META_PATH_WEIGHTS[n] for n in names],
        dtype=np.float64,
    )
    probs: np.ndarray = weights / weights.sum()
    choice: str = rng.choice(names, p=probs)
    return choice, META_PATHS[choice]


def generate_walks_for_partition(
    iterator,
    adj_dir: str,
    node_to_int: dict[str, int],
    r: int,
    target_len: int,
    seed: int,
):
    """mapPartitions callable: generate walks for one partition of songs.

    Each executor loads the adjacency Parquet files once, then
    generates *r* walks per song in its partition.
    """
    song_adj, node_adj = load_adjacency(adj_dir, node_to_int)

    for row in iterator:
        track_id: str = row.track_id
        if track_id not in node_to_int:
            continue
        song_int: int = node_to_int[track_id]

        for walk_id in range(r):
            walk_rng = np.random.default_rng(
                seed + song_int * 1000 + walk_id,
            )
            path_name, path_template = _choose_meta_path(walk_rng)

            walk_seq_ints: list[int] = follow_meta_path(
                song_int, path_template, song_adj, node_adj,
                walk_rng, target_len=target_len,
            )

            yield (
                str(track_id),
                int(walk_id),
                str(path_name),
                [int(x) for x in walk_seq_ints],
                int(len(walk_seq_ints)),
            )
