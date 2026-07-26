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
            pair_manifest = json.load(stream)
        if pair_manifest.get("stage") != "final_retrain":
            raise ValueError(f"{variant} pair manifest is not a retrain")
        pair_manifests[variant] = pair_manifest
    full_counts = pair_manifests["full"].get("counts", {})
    no_hard_counts = pair_manifests["no_hard_neg"].get("counts", {})
    if (
        pair_manifests["full"].get("query_count")
        != pair_manifests["no_hard_neg"].get("query_count")
        or full_counts.get("positive_count") != no_hard_counts.get("positive_count")
        or full_counts.get("negative_count") != no_hard_counts.get("negative_count")
        or float(
            pair_manifests["no_hard_neg"].get(
                "candidate_aware_target_fraction", -1.0
            )
        )
        != 0.0
    ):
        raise ValueError("No-hard-neg does not preserve the Full query/pair budget")
    protocol = create_set_c_protocol(
        args.output,
        scope=args.scope,
        parent_paths={
            "split_manifest": paths.split_manifest,
            "split_assignments": paths.split_assignments,
            "candidate_policy_manifest": paths.candidate_policy,
            "validation_group_thresholds": paths.validation_group_thresholds,
            "ranker_training_manifest": paths.ranker_training_manifest,
            "no_hard_neg_training_manifest": paths.no_hard_neg_training_manifest,
            "audio_index_manifest": paths.audio_manifest,
            "graph_index_manifest": paths.graph_manifest,
            "tag_idf": paths.tag_idf,
            "songs_metadata": paths.songs_metadata,
            "graph_edges": paths.graph_edges,
        },
    )
    print(
        "set_c_protocol_frozen "
        f"version={protocol['artifact_version']} output={args.output}"
    )


if __name__ == "__main__":
    main()
