from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pyspark import StorageLevel
from pyspark.ml.feature import PCAModel, StandardScalerModel, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from artifacts import sha256_path
from columns import PREPARED_AUDIO_COLUMNS, TRACK_ID_COLUMN
from l1_stats import (
    bootstrap_hedges_g_ci,
    classify_validation,
    distribution,
    hedges_g,
    preservation_summary,
)
from preprocess import add_scalar_availability, apply_frozen_preprocess
from train_pca import (
    EMBEDDING_COLUMN,
    FEATURES_COLUMN,
    PCA_FEATURES_COLUMN,
    SCALED_FEATURES_COLUMN,
    add_normalized_embedding,
)
from validate import read_metadata, require, validate_layout, validate_metadata


VALIDATION_VERSION = "c1_l1_1_v1"
PAIR_TYPES = ("same_artist", "same_release", "random")
METADATA_COLUMNS = (
    TRACK_ID_COLUMN,
    "song_id",
    "artist_id",
    "release_7digitalid",
    "year",
    "has_year",
)


def collect_scores(pairs: DataFrame, vectors: DataFrame, total_pairs: int) -> tuple[list[Any], int]:
    rows = score_pairs(pairs, vectors).collect()
    scored_count = len(rows)
    require(scored_count == total_pairs, "not every selected pair was scored")
    require(
        all(
            row[name] is not None and math.isfinite(float(row[name]))
            for row in rows
            for name in ("pre_pca_cosine", "pca_128_cosine")
        ),
        "L1-1 produced a non-finite cosine",
    )
    return rows, scored_count


def build_report(
    args: argparse.Namespace,
    encoder_metadata_path: Path,
    encoder_metadata: dict[str, Any],
    pair_counts: dict[str, int],
    distributions: dict[str, Any],
    effects: dict[str, Any],
    diagnostics: dict[str, Any],
    output_rows: int,
    base_rows: int,
    selected_count: int,
    scored_count: int,
    maximum_difference: float,
) -> dict[str, Any]:
    pair_requirement_met = all(
        pair_counts.get(pair_type) == args.pair_count for pair_type in PAIR_TYPES
    )
    finding = conclusion(effects, pair_requirement_met and not args.allow_partial_pairs)
    formal, status = classify_validation(
        pair_counts,
        PAIR_TYPES,
        args.pair_count,
        args.allow_partial_pairs,
        finding["supported"] is True,
    )
    return {
        "artifact_type": "c1_l1_feature_sanity_report",
        "validation_version": VALIDATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": status,
        "formal_pair_requirement_met": pair_requirement_met,
        "parameters": {
            "target_pairs_per_type": args.pair_count,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_method": "independent percentile bootstrap of Hedges' g",
            "confidence_level": 0.95,
            "seed": args.seed,
            "allow_partial_pairs": args.allow_partial_pairs,
            "random_matching": "exact year_key and feature-availability count",
        },
        "inputs": {
            "raw_input": str(args.raw_input.resolve()),
            "songs_metadata": str(args.songs_metadata.resolve()),
            "audio_output": str(args.output.resolve()),
            "audio_encoder_metadata_sha256": sha256_path(encoder_metadata_path),
            "shared_audio_contract_version": encoder_metadata["shared_audio_contract_version"],
            "c1_feature_version": encoder_metadata["c1_feature_version"],
        },
        "integrity": {
            "embedding_rows": output_rows,
            "eligible_rows": base_rows,
            "selected_tracks": selected_count,
            "scored_pairs": scored_count,
            "frozen_transform_max_abs_embedding_error": maximum_difference,
            "frozen_transform_tolerance": args.reproduction_tolerance,
        },
        "pair_counts": pair_counts,
        "distributions": distributions,
        "effect_size_vs_random": {
            "definition": "Hedges' g standardized mean difference; positive favors the relation",
            **effects,
        },
        **diagnostics,
        "conclusion": finding,
    }
