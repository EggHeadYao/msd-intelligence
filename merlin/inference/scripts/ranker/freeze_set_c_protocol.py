"""Freeze the Set-C evaluation inputs before labels are opened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...artifacts.paths import InferenceArtifactPaths
from ...artifacts.integrity import sha256_path
from ...evaluation.protocol import create_set_c_protocol


def parse_args() -> argparse.Namespace:
    defaults = InferenceArtifactPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=defaults.set_c_protocol)
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = InferenceArtifactPaths()
    opened_outputs = (
        paths.set_c_candidate_pool,
        paths.set_c_candidate_pool_manifest,
        paths.set_c_positives,
        paths.set_c_validation_pairs,
        paths.set_c_groups_manifest,
        paths.set_c_raw_features,
        paths.set_c_raw_features_manifest,
        paths.set_c_evaluation_report,
    )
    existing = [str(path) for path in opened_outputs if path.exists()]
    if existing:
        raise ValueError(
            f"cannot freeze a new protocol after Set-C outputs exist: {existing}"
        )
    manifests = {}
    for variant, path in (
        ("full", paths.ranker_training_manifest),
        ("no_hard_neg", paths.no_hard_neg_training_manifest),
    ):
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        actual_variant = manifest.get("selection", {}).get(
            "training_variant", "full"
        )
        if (
            manifest.get("artifact_type") != "ranker_training"
            or manifest.get("stage") != "final_retrain"
            or manifest.get("scope") != args.scope
            or manifest.get("converged") is not True
            or actual_variant != variant
        ):
            raise ValueError(f"cannot freeze invalid {variant} model")
        manifests[variant] = manifest
    if manifests["full"]["selected_reg_param"] != manifests["no_hard_neg"][
        "selected_reg_param"
    ]:
        raise ValueError("Full and no-hard-neg must use the same frozen regParam")
    if sha256_path(paths.ranker_scaler) != sha256_path(paths.no_hard_neg_scaler):
        raise ValueError("Full and no-hard-neg must share the frozen scaler")
    pair_manifests = {}
    for variant, path in (
        ("full", paths.training_pairs_manifest),
        ("no_hard_neg", paths.no_hard_neg_pairs_manifest),
    ):
        with path.open("r", encoding="utf-8") as stream:
