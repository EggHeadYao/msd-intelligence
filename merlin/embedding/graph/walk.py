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


def _load_adjacency_table(
    dataset_path: Path,
    name: str,
    type_codes: np.ndarray,
) -> AdjacencyTable:
    import pyarrow.parquet as pq

    arrow_table = pq.read_table(dataset_path)
    expected_columns = ("node_id", "neighbor_ids", "weights")
    if tuple(arrow_table.column_names) != expected_columns:
        raise ValueError(
            f"Invalid {name} columns: expected={expected_columns}, "
            f"actual={tuple(arrow_table.column_names)}",
        )

    node_ids = (
        arrow_table["node_id"]
        .to_numpy(zero_copy_only=False)
        .astype(
            np.int32,
            copy=False,
        )
    )
    if np.unique(node_ids).size != node_ids.size:
        raise ValueError(f"Duplicate {name} source node")

    source_type, neighbor_type = _ADJACENCY_TYPES[name]
    source_code = _NODE_TYPE_CODE[source_type]
    neighbor_code = _NODE_TYPE_CODE[neighbor_type]
    invalid_source = (node_ids < 0) | (node_ids >= len(type_codes))
    if invalid_source.any() or not (type_codes[node_ids] == source_code).all():
        raise ValueError(f"Invalid {name} source node")

    neighbor_blobs = arrow_table["neighbor_ids"].to_pylist()
    weight_blobs = arrow_table["weights"].to_pylist()
    neighbor_lengths = np.asarray(
        [len(value) // 4 if value is not None else -1 for value in neighbor_blobs],
        dtype=np.int64,
    )
    weight_lengths = np.asarray(
        [len(value) // 4 if value is not None else -1 for value in weight_blobs],
        dtype=np.int64,
    )
    malformed_binary = any(
        value is None or len(value) % 4 != 0 for value in neighbor_blobs
    ) or any(value is None or len(value) % 4 != 0 for value in weight_blobs)
    if (
        malformed_binary
        or (neighbor_lengths <= 0).any()
        or not np.array_equal(neighbor_lengths, weight_lengths)
    ):
        raise ValueError(f"Misaligned {name} paired binary arrays")

    offsets = np.empty(node_ids.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(neighbor_lengths, out=offsets[1:])
    neighbor_ids = np.frombuffer(b"".join(neighbor_blobs), dtype="<i4")
    weights = np.frombuffer(b"".join(weight_blobs), dtype="<f4")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError(f"Invalid {name} weights")
    invalid_neighbor = (neighbor_ids < 0) | (neighbor_ids >= len(type_codes))
    if invalid_neighbor.any() or not (type_codes[neighbor_ids] == neighbor_code).all():
        raise ValueError(f"Invalid {name} neighbor")

    row_by_node = np.full(len(type_codes), -1, dtype=np.int32)
    row_by_node[node_ids] = np.arange(node_ids.size, dtype=np.int32)
    return AdjacencyTable(row_by_node, offsets, neighbor_ids, weights)


def load_adjacency(
    adj_dir: str,
    node_to_int: dict[str, int],
) -> AdjacencyIndex:
    """Load and validate the seven paired adjacency datasets once per worker."""
    local_dir = _local_path(adj_dir).resolve()
    cache_key = (str(local_dir), id(node_to_int))
    cached = _ADJACENCY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    type_codes = _validate_vocabulary(node_to_int)
    adjacency: AdjacencyIndex = {}
    for name in ADJACENCY_NAMES:
        dataset_path = local_dir / f"{name}.parquet"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing C2 adjacency dataset: {dataset_path}")

        adjacency[name] = _load_adjacency_table(dataset_path, name, type_codes)

    _ADJACENCY_CACHE[cache_key] = adjacency
    return adjacency


def _neighbor_entry(
    adjacency: AdjacencyIndex,
    name: str,
    node_id: int,
) -> AdjacencyEntry:
    entry = adjacency[name].get(node_id)
    if entry is None:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    return entry


def _nodes_with_rows(
    values: np.ndarray,
    table: AdjacencyTable,
    *,
    excluded: int | None = None,
) -> np.ndarray:
    unique = np.unique(values)
    in_range = (unique >= 0) & (unique < table.row_by_node.size)
    unique = unique[in_range]
    active = unique[table.row_by_node[unique] >= 0]
    if excluded is not None:
        active = active[active != excluded]
    return active.astype(np.int32, copy=False)


def artist_transition_options(
    source_artist: int,
    adjacency: AdjacencyIndex,
) -> ArtistOptions:
    """Compute reusable P1-P3 choices for one source artist."""
    tracks, _ = _neighbor_entry(adjacency, "artist_to_tracks", source_artist)
    similar_artists, _ = _neighbor_entry(
        adjacency,
        "artist_to_similar_artists",
        source_artist,
    )
    active_similar = _nodes_with_rows(
        similar_artists,
        adjacency["artist_to_tracks"],
        excluded=source_artist,
    )

    terms, weights = _neighbor_entry(adjacency, "artist_to_terms", source_artist)
    eligible = np.zeros(terms.size, dtype=bool)
    for index, term_value in enumerate(terms):
        target_artists, _ = _neighbor_entry(
            adjacency,
            "term_to_artists",
            int(term_value),
        )
        eligible[index] = np.any(target_artists != source_artist)
    return ArtistOptions(
        tracks=tracks,
        similar_artists=active_similar,
        terms=terms[eligible],
        term_weights=weights[eligible],
    )


def eligible_transition_options(
    current_track: int,
    adjacency: AdjacencyIndex,
    artist_options_getter: Callable[[int], ArtistOptions] | None = None,
) -> TransitionOptions:
    """Compute every fully reachable path before selecting a path."""
    source_artists, _ = _neighbor_entry(adjacency, "track_to_artist", current_track)
    if source_artists.size > 1:
        raise ValueError(f"Track {current_track} has multiple artist neighbors")

    source_artist = int(source_artists[0]) if source_artists.size else -1
    if source_artist >= 0:
        artist_options = (
            artist_options_getter(source_artist)
            if artist_options_getter is not None
            else artist_transition_options(source_artist, adjacency)
        )
        p1_tracks = artist_options.tracks
        p2_artists = artist_options.similar_artists
        p3_terms = artist_options.terms
        p3_weights = artist_options.term_weights
    else:
        p1_tracks = np.empty(0, dtype=np.int32)
        p2_artists = np.empty(0, dtype=np.int32)
        p3_terms = np.empty(0, dtype=np.int32)
        p3_weights = np.empty(0, dtype=np.float32)

    releases, _ = _neighbor_entry(adjacency, "track_to_release", current_track)
    p4_releases: list[int] = []
    for release_value in releases:
        release_tracks, _ = _neighbor_entry(
            adjacency,
            "release_to_tracks",
            int(release_value),
        )
        if np.any(release_tracks != current_track):
            p4_releases.append(int(release_value))

    return TransitionOptions(
        current_track=current_track,
        source_artist=source_artist,
        p1_tracks=p1_tracks,
        p2_artists=p2_artists,
        p3_terms=p3_terms,
        p3_weights=p3_weights,
        p4_releases=np.asarray(p4_releases, dtype=np.int32),
    )


def _pick_uniform(values: np.ndarray, rng: np.random.Generator) -> int:
    return int(values[int(rng.integers(0, values.size))])


def _pick_excluding(
    values: np.ndarray,
    excluded: int,
    rng: np.random.Generator,
) -> int:
    position = int(np.searchsorted(values, excluded))
    present = position < values.size and int(values[position]) == excluded
    if not present:
        return _pick_uniform(values, rng)
    choice = int(rng.integers(0, values.size - 1))
    return int(values[choice + (choice >= position)])


def _pick_weighted_index(weights: np.ndarray, rng: np.random.Generator) -> int:
    total = float(weights.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        return int(rng.integers(0, weights.size))
    return int(rng.choice(weights.size, p=weights.astype(np.float64) / total))


def sample_transition(
    path_name: str,
    options: TransitionOptions,
    adjacency: AdjacencyIndex,
    rng: np.random.Generator,
) -> int:
    """Sample one endpoint using the path-specific hierarchical policy."""
    if path_name == "P1":
        return _pick_excluding(options.p1_tracks, options.current_track, rng)
    if path_name == "P2":
        artist = _pick_uniform(options.p2_artists, rng)
        tracks, _ = _neighbor_entry(adjacency, "artist_to_tracks", artist)
        return _pick_uniform(tracks, rng)
    if path_name == "P3":
        term_index = _pick_weighted_index(options.p3_weights, rng)
        term = int(options.p3_terms[term_index])
        artists, _ = _neighbor_entry(adjacency, "term_to_artists", term)
        artist = _pick_excluding(artists, options.source_artist, rng)
        tracks, _ = _neighbor_entry(adjacency, "artist_to_tracks", artist)
        return _pick_uniform(tracks, rng)
    if path_name == "P4":
        release = _pick_uniform(options.p4_releases, rng)
        tracks, _ = _neighbor_entry(adjacency, "release_to_tracks", release)
        return _pick_excluding(tracks, options.current_track, rng)
    raise ValueError(f"Unknown C2 path: {path_name}")


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
