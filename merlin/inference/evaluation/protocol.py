"""Reproducible protocol artifact for repeatable development evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

from ..artifacts.integrity import sha256_path
from ..artifacts.io import write_json_atomic


EVALUATION_VERSION = "merlin_development_evaluation_v1"
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


def create_development_protocol(
    output_path: str | Path,
    *,
    parent_paths: Mapping[str, str | Path],
    scope: str,
    evaluation_split: str = "set_c",
) -> dict[str, object]:
    """Bind the inputs and metrics for a reproducible development run."""
    if scope not in {"formal", "smoke"}:
        raise ValueError("development evaluation scope must be formal or smoke")
    if evaluation_split != "set_c":
        raise ValueError("development evaluation must use set_c")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"development protocol already exists: {output}")
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
        raise ValueError("development protocol parent set is incomplete or unexpected")
    payload = {
        "artifact_type": "development_evaluation_protocol",
        "artifact_version": EVALUATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "fit_split": "set_a",
        "selection_split": "set_b",
        "evaluation_split": evaluation_split,
        "retrain_splits": ["set_a", "set_b", "set_c", "remaining"],
        "seed": EVALUATION_SEED,
        "cutoffs": list(EVALUATION_CUTOFFS),
        "primary_cutoff": PRIMARY_CUTOFF,
        "query_bootstrap_samples": QUERY_BOOTSTRAP_SAMPLES,
        "artist_bootstrap_samples": ARTIST_BOOTSTRAP_SAMPLES,
        "scorers": list(SCORERS),
        "robustness_configs": list(ROBUSTNESS_CONFIGS),
        "report_sections": list(REPORT_SECTIONS),
        "claims": "development_metric_on_known_data_not_unbiased_generalization",
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(payload, output)
    return payload


def load_development_protocol(
    path: str | Path,
    *,
    expected_scope: str,
    expected_split: str = "set_c",
    expected_parent_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    if protocol.get("artifact_type") != "development_evaluation_protocol":
        raise ValueError("development evaluation protocol artifact type mismatch")
    if protocol.get("artifact_version") != EVALUATION_VERSION:
        raise ValueError("development evaluation protocol version mismatch")
    if protocol.get("scope") != expected_scope:
        raise ValueError("development evaluation protocol scope mismatch")
    boundary = (
        protocol.get("fit_split"),
        protocol.get("selection_split"),
        protocol.get("evaluation_split"),
    )
    if boundary != ("set_a", "set_b", expected_split):
        raise ValueError("development evaluation split boundary mismatch")
    if protocol.get("retrain_splits") != [
        "set_a", "set_b", "set_c", "remaining"
    ]:
        raise ValueError("development retrain universe mismatch")
    if protocol.get("cutoffs") != list(EVALUATION_CUTOFFS):
        raise ValueError("development evaluation cutoffs changed")
    if protocol.get("primary_cutoff") != PRIMARY_CUTOFF:
        raise ValueError("development primary cutoff changed")
    if protocol.get("scorers") != list(SCORERS):
        raise ValueError("development scorer set changed")
    if protocol.get("robustness_configs") != list(ROBUSTNESS_CONFIGS):
        raise ValueError("development robustness configuration changed")
    frozen_statistics = (
        protocol.get("seed") == EVALUATION_SEED
        and protocol.get("query_bootstrap_samples") == QUERY_BOOTSTRAP_SAMPLES
        and protocol.get("artist_bootstrap_samples") == ARTIST_BOOTSTRAP_SAMPLES
        and protocol.get("claims")
        == "development_metric_on_known_data_not_unbiased_generalization"
        and protocol.get("report_sections") == list(REPORT_SECTIONS)
    )
    if not frozen_statistics:
        raise ValueError("development statistical or report protocol changed")
    parents = protocol.get("parent_hashes")
    if not isinstance(parents, dict):
        raise ValueError("development evaluation protocol is missing parent hashes")
    for name, expected in (expected_parent_hashes or {}).items():
        if parents.get(name) != expected:
            raise ValueError(f"development protocol parent hash mismatch: {name}")
    return protocol
