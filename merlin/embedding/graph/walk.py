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
