"""Frozen candidate-generation policy for MERLIN inference."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping


CANDIDATE_POLICY_ARTIFACT_TYPE = "candidate_policy"
CANDIDATE_POLICY_MANIFEST_VERSION = "merlin_candidate_policy_v1"
CANDIDATE_POLICY_VERSION = "canonical-all-union-v1"
CANONICAL_RETRIEVER_LIMITS = {
    "audio": 250,
    "graph": 250,
    "bfs": 250,
    "tag": 250,
}
CANONICAL_CANDIDATE_LIMIT = 1_000
CANONICAL_LIMIT = 20
CANONICAL_VECTOR_OVERFETCH_FACTOR = 3
CANONICAL_BFS_MAX_DEPTH = 2
CANONICAL_BFS_PER_ARTIST_CAP = 10
CANONICAL_TAG_ARTIST_NEIGHBOR_LIMIT = 100
CANONICAL_TAG_MAX_TERM_ARTISTS = 5_000
CANONICAL_TAG_PER_ARTIST_CAP = 5


def canonical_policy_manifest() -> dict[str, object]:
    """Return the complete frozen Stage-1 policy artifact."""
    return {
        "artifact_type": CANDIDATE_POLICY_ARTIFACT_TYPE,
        "manifest_version": CANDIDATE_POLICY_MANIFEST_VERSION,
        "policy_version": CANDIDATE_POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "retriever_limits": dict(CANONICAL_RETRIEVER_LIMITS),
        "candidate_limit": CANONICAL_CANDIDATE_LIMIT,
        "final_limit": CANONICAL_FINAL_LIMIT,
        "same_song_filter": True,
        "vector": {
            "index_type": "IndexFlatIP",
            "overfetch_factor": CANONICAL_VECTOR_OVERFETCH_FACTOR,
        },
        "bfs": {
            "directed": True,
            "max_depth": CANONICAL_BFS_MAX_DEPTH,
            "per_artist_cap": CANONICAL_BFS_PER_ARTIST_CAP,
            "order": ["distance_ascending", "tag_cosine_descending", "track_id_ascending"],
        },
        "tag": {
            "artist_neighbor_limit": CANONICAL_TAG_ARTIST_NEIGHBOR_LIMIT,
            "max_term_artists": CANONICAL_TAG_MAX_TERM_ARTISTS,
            "per_artist_cap": CANONICAL_TAG_PER_ARTIST_CAP,
            "order": ["tfidf_cosine_descending", "artist_id_ascending", "track_id_ascending"],
        },
    }


def validate_canonical_policy(
    limits: Mapping[str, int],
    candidate_limit: int,
    final_limit: int,
) -> None:
    if dict(limits) != CANONICAL_RETRIEVER_LIMITS:
        raise ValueError("retriever limits do not match the canonical four-source policy")
    if candidate_limit != CANONICAL_CANDIDATE_LIMIT or final_limit != CANONICAL_FINAL_LIMIT:
        raise ValueError("candidate or final limit does not match the canonical policy")


def load_candidate_policy(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"candidate policy manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != CANDIDATE_POLICY_ARTIFACT_TYPE:
        raise ValueError("candidate policy artifact type mismatch")
    if manifest.get("manifest_version") != CANDIDATE_POLICY_MANIFEST_VERSION:
        raise ValueError("candidate policy manifest version mismatch")
    if manifest.get("policy_version") != CANDIDATE_POLICY_VERSION:
        raise ValueError("candidate policy version mismatch")
    limits = {str(name): int(value) for name, value in manifest.get("retriever_limits", {}).items()}
    validate_canonical_policy(
        limits,
        int(manifest.get("candidate_limit", -1)),
        int(manifest.get("final_limit", -1)),
    )
    expected = canonical_policy_manifest()
    for key in ("same_song_filter", "vector", "bfs", "tag"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"candidate policy {key} configuration mismatch")
    return manifest


def write_candidate_policy(path: str | Path) -> dict[str, object]:
    """Atomically write the frozen candidate policy."""
    manifest = canonical_policy_manifest()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)
    return manifest
