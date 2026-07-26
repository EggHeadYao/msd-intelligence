"""Evaluate frozen Full MERLIN and baselines once on canonical Set-C candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from ...artifacts.integrity import sha256_path
from ...artifacts.paths import InferenceArtifactPaths
from ...recall.pool import load_candidate_pool_manifest
from ...evaluation.metrics import (
    macro_metrics,
    paired_bootstrap_ci,
    random_ranking_expectation,
    score_query,
)
from ...evaluation.protocol import (
    ARTIST_BOOTSTRAP_SAMPLES,
    EVALUATION_CUTOFFS,
    EVALUATION_SEED,
    PRIMARY_CUTOFF,
    QUERY_BOOTSTRAP_SAMPLES,
    ROBUSTNESS_CONFIGS,
    SCORERS,
    load_set_c_protocol,
)
from ...artifacts.io import parquet_rows, read_row_artifact, write_json_atomic
from ...ranking.features import FEATURE_ORDER, FILL_FEATURES, load_raw_feature_manifest
from ...ranking.model import LogisticRanker
from ...training.validation_groups import (
    VALIDATION_QUERY_GROUPS,
    load_validation_group_manifest,
)


def parse_args() -> argparse.Namespace:
    paths = InferenceArtifactPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=paths.set_c_protocol)
    parser.add_argument("--candidate-pool", type=Path, default=paths.set_c_candidate_pool)
    parser.add_argument(
        "--candidate-pool-manifest",
        type=Path,
        default=paths.set_c_candidate_pool_manifest,
    )
    parser.add_argument("--groups-manifest", type=Path, default=paths.set_c_groups_manifest)
    parser.add_argument("--positives", type=Path, default=paths.set_c_positives)
    parser.add_argument("--validation-pairs", type=Path, default=paths.set_c_validation_pairs)
    parser.add_argument("--features", type=Path, default=paths.set_c_raw_features)
    parser.add_argument(
        "--features-manifest", type=Path, default=paths.set_c_raw_features_manifest
    )
    parser.add_argument("--output", type=Path, default=paths.set_c_evaluation_report)
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    return parser.parse_args()


def _protocol_parents(paths: InferenceArtifactPaths) -> dict[str, str]:
    return {
        "split_manifest": sha256_path(paths.split_manifest),
        "split_assignments": sha256_path(paths.split_assignments),
        "candidate_policy_manifest": sha256_path(paths.candidate_policy),
        "validation_group_thresholds": sha256_path(
            paths.validation_group_thresholds
        ),
        "ranker_training_manifest": sha256_path(paths.ranker_training_manifest),
        "no_hard_neg_training_manifest": sha256_path(
            paths.no_hard_neg_training_manifest
        ),
        "audio_index_manifest": sha256_path(paths.audio_manifest),
        "graph_index_manifest": sha256_path(paths.graph_manifest),
        "tag_idf": sha256_path(paths.tag_idf),
        "songs_metadata": sha256_path(paths.songs_metadata),
        "graph_edges": sha256_path(paths.graph_edges),
    }


def _load_ranker(root: Path, *, scope: str, variant: str) -> tuple[LogisticRanker, dict[str, float]]:
    schema = root / "ranker_feature_schema.json"
    scaler = root / "ranker_scaler.json"
    coefficients = root / "ranker_coefficients.json"
    manifest_path = root / "training_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("artifact_type") != "ranker_training"
        or manifest.get("scope") != scope
        or manifest.get("stage") != "final_retrain"
        or manifest.get("converged") is not True
    ):
        raise ValueError(f"{variant} ranker manifest is invalid")
    if manifest.get("selection", {}).get("training_variant", "full") != variant:
        raise ValueError(f"{variant} ranker training variant mismatch")
    hashes = manifest.get("artifact_hashes", {})
    for artifact in (schema, scaler, coefficients):
        if hashes.get(artifact.name) != sha256_path(artifact):
            raise ValueError(f"{variant} ranker artifact hash mismatch: {artifact.name}")
    with scaler.open("r", encoding="utf-8") as stream:
        scaler_payload = json.load(stream)
    fills = {name: float(value) for name, value in scaler_payload["fill_values"].items()}
    if set(fills) != set(FILL_FEATURES) or any(
        not math.isfinite(value) for value in fills.values()
    ):
        raise ValueError(f"{variant} ranker fill-value contract is invalid")
    ranker = LogisticRanker.from_artifacts(schema, scaler, coefficients)
    if ranker.feature_order != FEATURE_ORDER:
        raise ValueError(f"{variant} ranker feature order is not canonical")
    return ranker, fills


def _query_groups(path: Path) -> Iterable[tuple[str, list[dict[str, object]]]]:
    current_query = None
    current_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in read_row_artifact(path):
        query_id = str(row["query_track_id"])
        if current_query is None:
            current_query = query_id
        if query_id != current_query:
            if query_id in seen:
                raise ValueError("Set-C feature rows are not clustered by query")
            seen.add(current_query)
            yield current_query, current_rows
            current_query = query_id
            current_rows = []
        current_rows.append(row)
    if current_query is not None:
        if current_query in seen:
            raise ValueError("Set-C feature rows repeat a completed query")
        yield current_query, current_rows


def _candidate_metrics(rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    states = {
        group: {
