"""C2 graph embedding parameters and canonical graph schema."""

from __future__ import annotations

NUM_WALKS: int = 10
WALK_LENGTH: int = 40
SEED: int = 42

EDGE_SCHEMA: dict[str, tuple[str, str]] = {
    "track_artist": ("track", "artist"),
    "track_release": ("track", "release"),
    "artist_term": ("artist", "term"),
    "artist_similarity": ("artist", "artist"),
}

DIRECTED_EDGES: frozenset[str] = frozenset({"artist_similarity"})

META_PATHS: dict[str, list[str]] = {
    "P1": ["track_artist", "rev:track_artist"],
    "P2": ["track_artist", "artist_similarity", "rev:track_artist"],
    "P3": ["track_artist", "artist_term", "rev:artist_term", "rev:track_artist"],
    "P4": ["track_release", "rev:track_release"],
}

ADJACENCY_NAMES: tuple[str, ...] = (
    "track_to_artist",
    "artist_to_tracks",
    "track_to_release",
    "release_to_tracks",
    "artist_to_terms",
    "term_to_artists",
    "artist_to_similar_artists",
)
