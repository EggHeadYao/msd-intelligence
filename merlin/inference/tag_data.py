"""Runtime data generation for shared-tag candidate recall."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TagData:
    """Compact song ownership and bidirectional artist-tag mappings."""

    track_to_artist: Mapping[str, str]
    artist_tracks: Mapping[str, Sequence[str]]
    artist_terms: Mapping[str, frozenset[str]]
    term_artists: Mapping[str, frozenset[str]]


def build_tag_data(
    songs: Iterable[tuple[str, str]],
    terms: Iterable[tuple[str, str]],
) -> TagData:
    """Build duplicate-free mappings from projected metadata rows."""
    track_to_artist: dict[str, str] = {}
    artist_tracks: dict[str, list[str]] = defaultdict(list)
    seen_tracks: set[str] = set()
    for track_id, artist_id in songs:
        if not track_id or not artist_id:
            continue
        previous = track_to_artist.setdefault(track_id, artist_id)
        if previous != artist_id:
            raise ValueError(f"track {track_id!r} has multiple artists")
        if track_id not in seen_tracks:
            artist_tracks[artist_id].append(track_id)
            seen_tracks.add(track_id)

    artist_terms: dict[str, set[str]] = defaultdict(set)
    term_artists: dict[str, set[str]] = defaultdict(set)
    for artist_id, term in terms:
        if not artist_id or not term:
            continue
        normalized = term.strip().lower()
        if normalized:
            artist_terms[artist_id].add(normalized)
            term_artists[normalized].add(artist_id)

    return TagData(
        track_to_artist=track_to_artist,
        artist_tracks=dict(artist_tracks),
        artist_terms={key: frozenset(value) for key, value in artist_terms.items()},
        term_artists={key: frozenset(value) for key, value in term_artists.items()},
    )
