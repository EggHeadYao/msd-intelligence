"""CLI for positive-aware Candidate Recall attribution."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...artifact_paths import (
    CANDIDATE_AUDIT_PATH,
    CANDIDATE_POOL_MANIFEST_PATH,
    CANDIDATE_POOL_PATH,
    WEAK_LABEL_THRESHOLDS_PATH,
    WEAK_POSITIVES_MANIFEST_PATH,
    WEAK_POSITIVES_PATH,
)
from ...candidate_pool import load_candidate_pool_manifest
from ...candidate_audit import audit_candidate_pool, write_candidate_audit
from ...training.weak_labels import load_weak_positive_manifest, load_weak_positives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, default=CANDIDATE_POOL_PATH)
    parser.add_argument("--candidate-pool-manifest", type=Path, default=CANDIDATE_POOL_MANIFEST_PATH)
    parser.add_argument("--weak-positives", type=Path, default=WEAK_POSITIVES_PATH)
    parser.add_argument("--weak-positives-manifest", type=Path, default=WEAK_POSITIVES_MANIFEST_PATH)
    parser.add_argument("--thresholds", type=Path, default=WEAK_LABEL_THRESHOLDS_PATH)
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--output", type=Path, default=CANDIDATE_AUDIT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_candidate_pool_manifest(
        args.candidate_pool_manifest,
        args.candidate_pool,
        expected_scope=args.scope,
    )
    load_weak_positive_manifest(
        args.weak_positives_manifest,
        args.weak_positives,
        args.thresholds,
        expected_scope=args.scope,
    )
    report = audit_candidate_pool(
        args.candidate_pool,
        load_weak_positives(args.weak_positives),
    )
    write_candidate_audit(
        report,
        args.output,
        candidate_pool_path=args.candidate_pool,
        weak_positives_path=args.weak_positives,
    )
    print(
        "candidate_audit_ready "
        f"queries={report['eligible_query_count']} "
        f"macro_union_recall={report['macro_union_recall']:.6f} "
        f"output={args.output}",
    )


if __name__ == "__main__":
    main()
