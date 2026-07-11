from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


EXPECTED_SONGS: int = 1_000_000
EXPECTED_ARTIST_SIMILARITY_EDGES: int = 2_201_916
EXPECTED_ARTIST_TAG_EDGES: int = 1_109_381
EXPECTED_KNOWN_YEAR_SONGS: int = 515_576
EXPECTED_OUTPUT_DIRS: frozenset[str] = frozenset(
    {
        "songs_metadata.parquet",
        "song_audio_features_raw.parquet",
        "song_terms.parquet",
        "graph_edges.parquet",
    },
)
REQUIRED_EDGE_TYPES: frozenset[str] = frozenset(
    {
        "song_artist",
        "song_album",
        "song_tag",
        "artist_tag",
        "song_year",
        "artist_similarity",
    },
)
SEGMENT_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [
        f"{prefix}_{stat}_{i}"
        for prefix in ("pitch", "timbre")
        for stat in ("mean", "std", "min", "max")
        for i in range(12)
    ]
    + [f"loudness_{stat}" for stat in ("mean", "std", "min", "max")]
)
AUDIO_SCALAR_COLUMNS: tuple[str, ...] = (
    "track_id",
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "duration",
    "key",
    "mode",
    "time_signature",
)
