"""Compact runtime mappings used by artist-graph BFS recall."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..artifacts.io import parquet_rows


@dataclass(frozen=True, slots=True)
class BfsData:
    """Only the metadata needed by ``BfsRetriever`` at query time."""

    track_to_artist: Mapping[str, str]
    artist_tracks: Mapping[str, Sequence[str]]
    artist_neighbors: Mapping[str, Sequence[str]]


def build_bfs_data(
    songs: Iterable[tuple[str, str]],
    edges: Iterable[tuple[str, str]],
) -> BfsData:
    """Build deterministic, duplicate-free mappings from projected rows."""
    track_to_artist: dict[str, str] = {}
    artist_tracks: dict[str, list[str]] = defaultdict(list)
    for track_id, artist_id in songs:
        if not track_id or not artist_id:
            continue
        previous = track_to_artist.setdefault(track_id, artist_id)
        if previous != artist_id:
            raise ValueError(f"track {track_id!r} has multiple artists")
        if track_id not in artist_tracks[artist_id]:
            artist_tracks[artist_id].append(track_id)

    artist_neighbors: dict[str, list[str]] = defaultdict(list)
    seen_edges: set[tuple[str, str]] = set()
    for source, target in edges:
        edge = (source, target)
        if not source or not target or source == target or edge in seen_edges:
            continue
        seen_edges.add(edge)
        artist_neighbors[source].append(target)

    return BfsData(
        track_to_artist=track_to_artist,
        artist_tracks=dict(artist_tracks),
        artist_neighbors=dict(artist_neighbors),
    )


def load_bfs_data(
    songs_metadata_path: str | Path,
    graph_edges_path: str | Path,
) -> BfsData:
    """Load metadata and directed artist-similarity edges without Spark."""
    songs = _parquet_rows(songs_metadata_path, ("track_id", "artist_id"))
    edges = _graph_edge_rows(graph_edges_path, "artist_similarity")
    return build_bfs_data(songs, edges)


def load_artist_neighbors(graph_edges_path: str | Path) -> Mapping[str, Sequence[str]]:
    """Load only the directed artist-similarity adjacency for shared assembly."""
    edges = _graph_edge_rows(graph_edges_path, "artist_similarity")
    return build_bfs_data((), edges).artist_neighbors


def _graph_edge_rows(path: str | Path, edge_type: str) -> Iterable[tuple[str, str]]:
    yield from parquet_rows(path, ("src_id", "dst_id"), edge_type=edge_type)


def _parquet_rows(
    path: str | Path,
    columns: tuple[str, str],
) -> Iterable[tuple[str, str]]:
    yield from parquet_rows(path, columns)
