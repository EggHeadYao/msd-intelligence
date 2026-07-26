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
        "scope": scope,
        "fit_split": "set_a",
        "selection_split": "set_b",
        "evaluation_split": "set_c",
        "seed": EVALUATION_SEED,
        "cutoffs": list(EVALUATION_CUTOFFS),
        "primary_cutoff": PRIMARY_CUTOFF,
        "query_bootstrap_samples": QUERY_BOOTSTRAP_SAMPLES,
        "artist_bootstrap_samples": ARTIST_BOOTSTRAP_SAMPLES,
        "scorers": list(SCORERS),
        "robustness_configs": list(ROBUSTNESS_CONFIGS),
        "report_sections": list(REPORT_SECTIONS),
        "claims": "weak_label_task_consistency_not_user_relevance",
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(payload, output)
    return payload


def load_set_c_protocol(
    path: str | Path,
    *,
    expected_scope: str,
    expected_parent_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    if protocol.get("artifact_type") != "set_c_evaluation_protocol":
        raise ValueError("Set-C evaluation protocol artifact type mismatch")
    if protocol.get("artifact_version") != EVALUATION_VERSION:
        raise ValueError("Set-C evaluation protocol version mismatch")
    if protocol.get("scope") != expected_scope:
        raise ValueError("Set-C evaluation protocol scope mismatch")
    boundary = (
        protocol.get("fit_split"),
        protocol.get("selection_split"),
        protocol.get("evaluation_split"),
    )
    if boundary != ("set_a", "set_b", "set_c"):
        raise ValueError("Set-C evaluation split boundary mismatch")
    if protocol.get("cutoffs") != list(EVALUATION_CUTOFFS):
        raise ValueError("Set-C evaluation cutoffs changed after freezing")
    if protocol.get("primary_cutoff") != PRIMARY_CUTOFF:
        raise ValueError("Set-C primary cutoff changed after freezing")
    if protocol.get("scorers") != list(SCORERS):
        raise ValueError("Set-C scorer set changed after freezing")
    if protocol.get("robustness_configs") != list(ROBUSTNESS_CONFIGS):
        raise ValueError("Set-C robustness configuration changed after freezing")
    frozen_statistics = (
        protocol.get("seed") == EVALUATION_SEED
        and protocol.get("query_bootstrap_samples") == QUERY_BOOTSTRAP_SAMPLES
        and protocol.get("artist_bootstrap_samples") == ARTIST_BOOTSTRAP_SAMPLES
        and protocol.get("claims") == "weak_label_task_consistency_not_user_relevance"
        and protocol.get("report_sections") == list(REPORT_SECTIONS)
    )
    if not frozen_statistics:
        raise ValueError("Set-C statistical or report protocol changed after freezing")
    parents = protocol.get("parent_hashes")
    if not isinstance(parents, dict):
        raise ValueError("Set-C evaluation protocol is missing parent hashes")
    for name, expected in (expected_parent_hashes or {}).items():
        if parents.get(name) != expected:
            raise ValueError(f"Set-C protocol parent hash mismatch: {name}")
    return protocol
