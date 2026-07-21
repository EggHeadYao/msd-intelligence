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
_NODE_TYPE_CODE = {"track": 0, "artist": 1, "release": 2, "term": 3}


@dataclass(frozen=True)
class AdjacencyTable:
    """Compact row lookup plus CSR-style paired neighbor arrays."""

    row_by_node: np.ndarray
    offsets: np.ndarray
    neighbor_ids: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_entries(cls, entries: dict[int, AdjacencyEntry]) -> AdjacencyTable:
        if not entries:
            return cls(
                row_by_node=np.empty(0, dtype=np.int32),
                offsets=np.zeros(1, dtype=np.int64),
                neighbor_ids=np.empty(0, dtype=np.int32),
                weights=np.empty(0, dtype=np.float32),
            )
        node_ids = np.asarray(sorted(entries), dtype=np.int32)
        row_by_node = np.full(int(node_ids[-1]) + 1, -1, dtype=np.int32)
        row_by_node[node_ids] = np.arange(node_ids.size, dtype=np.int32)
        neighbor_parts = [entries[int(node_id)][0] for node_id in node_ids]
        weight_parts = [entries[int(node_id)][1] for node_id in node_ids]
        lengths = np.asarray([part.size for part in neighbor_parts], dtype=np.int64)
        offsets = np.empty(node_ids.size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        return cls(
            row_by_node=row_by_node,
            offsets=offsets,
            neighbor_ids=np.concatenate(neighbor_parts),
            weights=np.concatenate(weight_parts),
        )

    def get(self, node_id: int) -> AdjacencyEntry | None:
        if not 0 <= node_id < self.row_by_node.size:
            return None
        row = int(self.row_by_node[node_id])
        if row < 0:
            return None
        start = int(self.offsets[row])
        end = int(self.offsets[row + 1])
        return self.neighbor_ids[start:end], self.weights[start:end]


AdjacencyIndex = dict[str, AdjacencyTable]
_ADJACENCY_CACHE: dict[tuple[str, int], AdjacencyIndex] = {}


@dataclass(frozen=True)
class ArtistOptions:
    """Reusable path choices determined only by the source artist."""

    tracks: np.ndarray
    similar_artists: np.ndarray
    terms: np.ndarray
    term_weights: np.ndarray


@dataclass(frozen=True)
class TransitionOptions:
    """Eligible hierarchical choices for one track-to-track transition."""

    current_track: int
    source_artist: int
    p1_tracks: np.ndarray
    p2_artists: np.ndarray
    p3_terms: np.ndarray
    p3_weights: np.ndarray
    p4_releases: np.ndarray

    def eligible_paths(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, available in (
                ("P1", self.p1_tracks.size > 1),
                ("P2", self.p2_artists.size > 0),
                ("P3", self.p3_terms.size > 0),
                ("P4", self.p4_releases.size > 0),
            )
            if available
        )


def _local_path(path: str) -> Path:
    parsed = urlparse(path)
    if parsed.scheme not in {"", "file"}:
        raise ValueError(f"C2 adjacency requires a local path, got: {path}")
    return Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(path)


def _validate_vocabulary(node_to_int: dict[str, int]) -> np.ndarray:
    if not node_to_int:
        raise ValueError("C2 vocabulary is empty")

    type_codes = np.full(len(node_to_int), -1, dtype=np.int8)
    for typed_key, node_id in node_to_int.items():
        node_type, raw_id = decode_typed_key(typed_key)
        if encode_typed_key(node_type, raw_id) != typed_key:
            raise ValueError(f"Non-canonical typed vocabulary key: {typed_key!r}")
        if not isinstance(node_id, int) or not 0 <= node_id < len(type_codes):
            raise ValueError(f"Invalid vocabulary integer ID: {node_id!r}")
        if type_codes[node_id] != -1:
            raise ValueError(f"Duplicate vocabulary integer ID: {node_id}")
        if node_type not in _NODE_TYPE_CODE:
            raise ValueError(f"Invalid vocabulary node type: {node_type!r}")
        type_codes[node_id] = _NODE_TYPE_CODE[node_type]

    if (type_codes < 0).any():
        raise ValueError("C2 vocabulary integer IDs must be contiguous from zero")
    return type_codes


def clear_adjacency_cache() -> None:
    """Clear worker-local adjacency state, primarily for isolated tests."""
    _ADJACENCY_CACHE.clear()


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
