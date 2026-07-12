"""C2 graph embedding: meta-path random walk parameters and schema definitions."""

from __future__ import annotations

# -- Walk parameters ----------------------------------------------
NUM_WALKS: int = 3  # r: walks per song
WALK_LENGTH: int = 40  # L: target song nodes per walk sequence
SEED: int = 42

# -- Meta-path selection weights (proportional to signal quality) -
META_PATH_WEIGHTS: dict[str, float] = {
    "P1": 1.0,  # artist_similarity (strongest signal)
    "P2": 0.8,  # per-song similar_artist
    "P3": 0.6,  # per-song tag
    "P4": 0.4,  # artist-level tag (coarse)
    "P5": 0.8,  # same album
    "P6": 0.4,  # same year
}

# -- Meta-path definitions: ordered edge-type sequences -----------
# rev:X = reverse of edge type X (swap src<->dst).
# Undirected edges (directed=false) have bidirectional adjacency;
# Directed edges (artist_similarity, song_similar_artist) have
# forward-only adjacency -- rev:X is NOT available for them.

META_PATHS: dict[str, list[str]] = {
    "P1": ["song_artist", "artist_similarity", "rev:song_artist"],  # 3-hop
    "P2": ["song_similar_artist", "rev:song_artist"],  # 2-hop
    "P3": ["song_tag", "rev:song_tag"],  # 2-hop
    "P4": ["song_artist", "artist_tag", "rev:artist_tag", "rev:song_artist"],  # 4-hop
    "P5": ["song_album", "rev:song_album"],  # 2-hop
    "P6": ["song_year", "rev:song_year"],  # 2-hop
}

# -- Edge type -> (src_type, dst_type) mapping --------------------
EDGE_SCHEMA: dict[str, tuple[str, str]] = {
    "song_artist": ("song", "artist"),
    "song_album": ("song", "album"),
    "song_tag": ("song", "tag"),
    "artist_tag": ("artist", "tag"),
    "song_year": ("song", "year"),
    "artist_similarity": ("artist", "artist"),
    "song_similar_artist": ("song", "artist"),
}

# -- Directed edge types (reverse NOT stored in adjacency index) --
DIRECTED_EDGES: frozenset[str] = frozenset({"artist_similarity", "song_similar_artist"})
