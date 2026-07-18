"""Lineage-aware loader for the split Ranker artifact bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .artifact_lineage import sha256_path
from .feature_schema import RANKER_V2_FEATURES, RANKER_V2_SCHEMA_VERSION
from .ranker import LogisticRanker


def load_ranker_bundle(
    schema_path: str | Path,
    scaler_path: str | Path,
    coefficients_path: str | Path,
    training_manifest_path: str | Path,
    *,
    expected_parent_hashes: Mapping[str, str],
) -> LogisticRanker:
    """Validate one inseparable Set-A Ranker bundle and return its scorer."""
    paths = tuple(Path(path) for path in (schema_path, scaler_path, coefficients_path))
    manifest_path = Path(training_manifest_path)
    for path in (*paths, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"ranker artifact does not exist: {path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("feature_schema_version") != RANKER_V2_SCHEMA_VERSION:
        raise ValueError("Ranker training manifest schema version mismatch")
    if tuple(manifest.get("feature_order", ())) != RANKER_V2_FEATURES:
        raise ValueError("Ranker training manifest feature order mismatch")

    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("Ranker training manifest is missing artifact hashes")
    for path in paths:
        if artifact_hashes.get(path.name) != sha256_path(path):
            raise ValueError(f"Ranker artifact hash mismatch: {path.name}")

    parent_hashes = manifest.get("parent_hashes")
    if not isinstance(parent_hashes, dict):
        raise ValueError("Ranker training manifest is missing parent hashes")
    for name, expected in expected_parent_hashes.items():
        if parent_hashes.get(name) != expected:
            raise ValueError(f"Ranker parent hash mismatch: {name}")

    ranker = LogisticRanker.from_artifacts(*paths)
    if ranker.feature_schema_version != RANKER_V2_SCHEMA_VERSION:
        raise ValueError("Ranker scorer schema version mismatch")
    if ranker.feature_order != RANKER_V2_FEATURES:
        raise ValueError("Ranker scorer feature order mismatch")
    return ranker
