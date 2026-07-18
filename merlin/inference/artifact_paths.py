"""Canonical MERLIN v2 artifact locations used by pure-Python inference."""

from pathlib import Path


PARQUETS_ROOT = Path("parquets_new")
PREPARED_ROOT = PARQUETS_ROOT / "prepared"
MERLIN_ROOT = PARQUETS_ROOT / "merlin"
AUDIO_ROOT = MERLIN_ROOT / "audio"
GRAPH_ROOT = MERLIN_ROOT / "graph"
RANKER_ROOT = MERLIN_ROOT / "ranker"

AUDIO_INDEX_PATH = AUDIO_ROOT / "index_audio.faiss"
AUDIO_MAPPING_PATH = AUDIO_ROOT / "index_audio_track_ids.parquet"
AUDIO_MANIFEST_PATH = AUDIO_ROOT / "index_audio_manifest.json"

GRAPH_INDEX_PATH = GRAPH_ROOT / "index_graph.faiss"
GRAPH_MAPPING_PATH = GRAPH_ROOT / "index_graph_track_ids.parquet"
GRAPH_MANIFEST_PATH = GRAPH_ROOT / "index_graph_manifest.json"

SONGS_METADATA_PATH = PREPARED_ROOT / "songs_metadata.parquet"
GRAPH_EDGES_PATH = PREPARED_ROOT / "graph_edges.parquet"

RANKER_SCHEMA_PATH = RANKER_ROOT / "ranker_feature_schema.json"
RANKER_SCALER_PATH = RANKER_ROOT / "ranker_scaler.json"
RANKER_COEFFICIENTS_PATH = RANKER_ROOT / "ranker_coefficients.json"
CANDIDATE_POLICY_PATH = RANKER_ROOT / "candidate_policy_manifest.json"
