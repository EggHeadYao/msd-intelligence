"""Fail-closed protocol artifact for the one-time Set-C evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

from ..artifacts.integrity import sha256_path
from ..artifacts.io import write_json_atomic


EVALUATION_VERSION = "merlin_set_c_evaluation_v1"
EVALUATION_CUTOFFS = (10, 20, 50)
PRIMARY_CUTOFF = 20
QUERY_BOOTSTRAP_SAMPLES = 2_000
ARTIST_BOOTSTRAP_SAMPLES = 1_000
EVALUATION_SEED = 42
SCORERS = (
    "full",
    "random",
    "c1_only",
    "c2_only",
    "handcrafted",
    "bfs",
    "no_hard_neg",
)
ROBUSTNESS_CONFIGS = ("precomputed_acoustic_cold",)
REPORT_SECTIONS = (
    "group_eligibility",
    "candidate_layer",
    "ranking",
    "paired_inference_full_minus_baseline",
    "robustness",
    "lineage",
)


def create_set_c_protocol(
    output_path: str | Path,
    *,
    parent_paths: Mapping[str, str | Path],
    scope: str,
) -> dict[str, object]:
    """Freeze every input and metric choice before any Set-C label is opened."""
    if scope not in {"formal", "smoke"}:
        raise ValueError("Set-C evaluation scope must be formal or smoke")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Set-C evaluation protocol already exists: {output}")
    required = {
        "split_manifest",
        "split_assignments",
        "candidate_policy_manifest",
        "validation_group_thresholds",
        "ranker_training_manifest",
        "no_hard_neg_training_manifest",
        "audio_index_manifest",
        "graph_index_manifest",
        "tag_idf",
        "songs_metadata",
        "graph_edges",
    }
    if set(parent_paths) != required:
        raise ValueError("Set-C protocol parent set is incomplete or unexpected")
    payload = {
        "artifact_type": "set_c_evaluation_protocol",
        "artifact_version": EVALUATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
