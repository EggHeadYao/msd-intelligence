"""Frozen candidate-generation policy for MERLIN inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


CANDIDATE_POLICY_VERSION = "canonical-all-union-v1"
CANONICAL_RETRIEVER_LIMITS = {
    "audio": 250,
    "graph": 250,
    "bfs": 250,
    "tag": 250,
}
CANONICAL_CANDIDATE_LIMIT = 1_000
CANONICAL_FINAL_LIMIT = 20


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
    if manifest.get("policy_version") != CANDIDATE_POLICY_VERSION:
        raise ValueError("candidate policy version mismatch")
    limits = {str(name): int(value) for name, value in manifest.get("retriever_limits", {}).items()}
    validate_canonical_policy(
        limits,
        int(manifest.get("candidate_limit", -1)),
        int(manifest.get("final_limit", -1)),
    )
    return manifest
