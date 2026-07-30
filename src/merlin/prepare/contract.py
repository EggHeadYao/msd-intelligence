"""Stable schemas and counts for MERLIN prepared tables."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from tools.hdf5.audio_features import (
    CONTRACT_VERSION,
    FEATURE_COLUMNS,
    MERLIN_EXCLUDED_COLUMNS,
    MERLIN_FEATURE_COLUMNS,
)


ARTIFACT_TYPE = "prepared_tables"
ARTIFACT_VERSION = "v2"
MANIFEST_NAME = "prepared_manifest.json"
FEATURE_CONTRACT_NAME = "feature_contract.json"
SHARED_AUDIO_CONTRACT_VERSION = CONTRACT_VERSION
SHARED_AUDIO_FEATURE_COUNT = len(FEATURE_COLUMNS)
MERLIN_AUDIO_FEATURE_COUNT = len(MERLIN_FEATURE_COLUMNS)
MERLIN_RAW_FEATURE_COUNT = 11 + MERLIN_AUDIO_FEATURE_COUNT
FEATURE_ORDER_SHA256 = hashlib.sha256(
    json.dumps(
        FEATURE_COLUMNS,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii"),
).hexdigest()

if SHARED_AUDIO_FEATURE_COUNT != 628:
    raise RuntimeError("Shared audio contract must contain 628 features")
if len(MERLIN_EXCLUDED_COLUMNS) != 76 or MERLIN_AUDIO_FEATURE_COUNT != 552:
    raise RuntimeError("MERLIN projection must exclude 76 of 628 features")
if MERLIN_RAW_FEATURE_COUNT != 563:
    raise RuntimeError("MERLIN raw view must contain 563 features")

OUTPUT_DIRS: frozenset[str] = frozenset(
    {
        "songs_metadata.parquet",
        "song_audio_features_raw.parquet",
        "graph_edges.parquet",
    },
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "track_id",
    "loudness",
    "tempo",
    "duration",
    "key",
    "key_confidence",
    "mode",
    "mode_confidence",
    "time_signature",
    "time_signature_confidence",
    "end_of_fade_in",
    "start_of_fade_out",
    "artist_id",
    "artist_name",
    "release",
    "release_7digitalid",
    "song_id",
    "song_hotttnesss",
    "artist_hotttnesss",
    "artist_familiarity",
    "title",
    "track_7digitalid",
    "year",
)

TRACK_METADATA_COLUMNS: tuple[str, ...] = (
    "track_id",
    "title",
    "song_id",
    "release",
    "artist_id",
    "artist_mbid",
    "artist_name",
    "duration",
    "artist_familiarity",
    "artist_hotttnesss",
    "year",
)

ARTIST_TERM_COLUMNS: tuple[str, ...] = ("artist_id", "term")
ARTIST_SIMILARITY_COLUMNS: tuple[str, ...] = ("src", "dst")
EXTRACTED_AUDIO_COLUMNS: tuple[str, ...] = ("track_id", *FEATURE_COLUMNS)

SUMMARY_AUDIO_COLUMNS: tuple[str, ...] = (
    "track_id",
    "loudness",
    "tempo",
    "duration",
    "key",
    "key_confidence",
    "mode",
    "mode_confidence",
    "time_signature",
    "time_signature_confidence",
    "end_of_fade_in",
    "start_of_fade_out",
)

METADATA_COLUMNS: tuple[str, ...] = (
    "track_id",
    "song_id",
    "title",
    "artist_id",
    "artist_name",
    "artist_mbid",
    "release",
    "release_7digitalid",
    "track_7digitalid",
    "duration",
    "year",
    "has_year",
    "song_hotttnesss",
    "artist_hotttnesss",
    "artist_familiarity",
)

AUDIO_COLUMNS: tuple[str, ...] = (
    *SUMMARY_AUDIO_COLUMNS,
    *MERLIN_FEATURE_COLUMNS,
)
GRAPH_EDGE_COLUMNS: tuple[str, ...] = (
    "src_type",
    "src_id",
    "dst_type",
    "dst_id",
    "directed",
    "edge_type",
)

EDGE_TYPES: tuple[str, ...] = (
    "track_artist",
    "track_release",
    "artist_term",
    "artist_similarity",
)
NODE_TYPES: frozenset[str] = frozenset({"track", "artist", "release", "term"})


@dataclass(frozen=True)
class ExpectedCounts:
    """Expected table and relation counts for one validation run."""

    songs: int = 1_000_000
    track_release: int = 999_997
    artist_term: int = 1_109_381
    artist_similarity: int = 2_201_916

    @property
    def graph_edges(self) -> int:
        return (
            self.songs + self.track_release + self.artist_term + self.artist_similarity
        )
