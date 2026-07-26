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
