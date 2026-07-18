"""Compact runtime mappings used by artist-graph BFS recall."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


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
