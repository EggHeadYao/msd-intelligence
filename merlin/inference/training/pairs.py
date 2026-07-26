"""Deterministic split-safe candidate-aware Ranker pair construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from heapq import nsmallest
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..artifacts.integrity import artifact_size_bytes, sha256_path
from ..recall.pool import iter_candidate_pool
from ..recall.streaming import EncodedCandidates
from ..artifacts.io import PartitionedParquetWriter, write_json_atomic, write_row_artifact
from ..ranking.features import (
    FEATURE_SCHEMA,
    RAW_BASE_FEATURES,
    RAW_FEATURE_VERSION,
    SAMPLE_WEIGHT_COLUMN,
    raw_feature_parquet_schema,
)
from ..types import Candidate
from .weak_labels import iter_weak_positives


TRAINING_PAIR_VERSION = "merlin_training_pairs_v2"
PAIR_SEED = 42
NEGATIVE_RATIO = 3
CANDIDATE_AWARE_FRACTION = 0.75
LOSS_WEIGHT_STRATEGY = "per_query_candidate_curriculum"
MAX_CANDIDATE_SAMPLE_WEIGHT = 20.0
IsPositive = Callable[[str, str], bool]
IsPositiveBatch = Callable[[str, Sequence[str]], Sequence[bool]]
IsPositivePairs = Callable[[Sequence[tuple[str, str]]], Sequence[bool]]
SameSong = Callable[[str, str], bool]
WeakPositiveMap = Mapping[str, Mapping[str, frozenset[str]]]
WeakPositiveSource = str | Path | WeakPositiveMap
CandidateInput = Mapping[str, object] | Candidate
CandidateCollection = Sequence[CandidateInput] | EncodedCandidates


@dataclass(frozen=True, slots=True)
class StreamCheckpoint:
    """A batch boundary at which both output datasets are recoverable."""

    processed_queries: int
    total_queries: int


@dataclass(frozen=True, slots=True)
class StreamTableBatch:
    """Aligned Arrow tables and per-query audit totals for one recall batch."""

    pairs: Any
    features: Any
    audits: Sequence[Mapping[str, object]]
    weak_source_totals: Mapping[str, int]
    recall_source_totals: Mapping[str, int]


def training_pair_parquet_schema():
    import pyarrow as pa

    return pa.schema((
        pa.field("query_track_id", pa.string(), nullable=False),
        pa.field("candidate_track_id", pa.string(), nullable=False),
        pa.field("label", pa.int64(), nullable=False),
        pa.field(SAMPLE_WEIGHT_COLUMN, pa.float32(), nullable=False),
        pa.field("positive_sources", pa.list_(pa.string()), nullable=False),
        pa.field("negative_source", pa.string()),
        pa.field("recall_sources", pa.list_(pa.string()), nullable=False),
    ))


def _negative_loss_weights(
    negative_count: int,
    candidate_count: int,
    random_count: int,
    candidate_target: int,
) -> tuple[float, float]:
    """Preserve 1:3 total loss while restoring the hard-negative target mix."""
    if negative_count != candidate_count + random_count or negative_count <= 0:
        raise ValueError("negative loss-weight counts are inconsistent")
    if not 0 <= candidate_target <= negative_count:
        raise ValueError("candidate loss-weight target is invalid")
    if candidate_count == 0:
        return 0.0, negative_count / random_count
    if random_count == 0:
        return negative_count / candidate_count, 0.0
    candidate_weight = min(
        candidate_target / candidate_count,
        MAX_CANDIDATE_SAMPLE_WEIGHT,
    )
    candidate_weight_sum = candidate_weight * candidate_count
    return candidate_weight, (negative_count - candidate_weight_sum) / random_count


def _histogram_percentiles(
    histogram: Mapping[str, int], probabilities: Mapping[str, float]
) -> dict[str, float | None]:
    count = sum(int(value) for value in histogram.values())
    if count == 0:
        return {name: None for name in probabilities}
    ordered = sorted(
        ((float(key), int(item)) for key, item in histogram.items())
    )
    result = {}
    for name, probability in probabilities.items():
        target = max(1, int(np.ceil(probability * count)))
        cumulative = 0
        for value, frequency in ordered:
            cumulative += frequency
            if cumulative >= target:
                result[name] = value
                break
        else:
            raise AssertionError("loss-weight histogram percentile is incomplete")
    return result


def _loss_weight_shape_key(audit: Mapping[str, object]) -> str:
    candidate_count = int(audit["candidate_aware_count"])
    candidate_target = candidate_count + int(audit["candidate_shortage"])
    return f'{int(audit["negative_count"])}:{candidate_count}:{candidate_target}'


def _loss_weight_histograms(
    stats: Mapping[str, object],
) -> tuple[Counter[str], Counter[str], int]:
    shapes = stats.get("loss_weight_shape_histogram")
    if shapes is None:
        return (
            Counter(stats.get("candidate_weight_histogram", {})),
            Counter(stats.get("effective_fraction_histogram", {})),
            int(stats.get("queries_without_candidate_aware", 0)),
        )
    weights: Counter[str] = Counter(
        stats.get("legacy_candidate_weight_histogram", {})
    )
    fractions: Counter[str] = Counter(
        stats.get("legacy_effective_fraction_histogram", {})
    )
    queries_without_candidate = int(
        stats.get("legacy_queries_without_candidate_aware", 0)
    )
    for key, frequency_value in dict(shapes).items():
        negative_count, candidate_count, candidate_target = (
            int(value) for value in str(key).split(":")
        )
        frequency = int(frequency_value)
        if candidate_count == 0:
            queries_without_candidate += frequency
            fractions["0"] += frequency
            continue
        candidate_weight, _ = _negative_loss_weights(
            negative_count,
            candidate_count,
            negative_count - candidate_count,
            candidate_target,
        )
        weights[format(candidate_weight, ".12g")] += candidate_count * frequency
        fractions[
            format(candidate_weight * candidate_count / negative_count, ".12g")
        ] += frequency
    return weights, fractions, queries_without_candidate


def _loss_weight_audit_manifest(stats: Mapping[str, object]) -> dict[str, object]:
    weight_histogram, fraction_histogram, queries_without_candidate = (
        _loss_weight_histograms(stats)
    )
    return {
        "candidate_weight_cap": MAX_CANDIDATE_SAMPLE_WEIGHT,
        "queries_without_candidate_aware": queries_without_candidate,
        "candidate_weight": _histogram_percentiles(
            weight_histogram,
            {"median": 0.50, "p95": 0.95, "p99": 0.99, "max": 1.0},
        ),
        "per_query_effective_candidate_aware_fraction": _histogram_percentiles(
            fraction_histogram,
            {"median": 0.50, "p95": 0.95, "p99": 0.99},
        ),
    }


def _loss_weight_manifest(
    totals: Mapping[str, float],
    candidate_target_fraction: float,
    stats: Mapping[str, object],
) -> dict[str, object]:
    negative = float(totals.get("candidate_aware", 0.0)) + float(
        totals.get("random", 0.0)
    )
    candidate = float(totals.get("candidate_aware", 0.0))
    return {
        "column": SAMPLE_WEIGHT_COLUMN,
        "strategy": LOSS_WEIGHT_STRATEGY,
        "candidate_aware_target_fraction": float(candidate_target_fraction),
        "positive_weight_sum": float(totals.get("positive", 0.0)),
        "candidate_aware_weight_sum": candidate,
        "random_weight_sum": float(totals.get("random", 0.0)),
        "negative_weight_sum": negative,
        "effective_candidate_aware_fraction": candidate / negative if negative else 0.0,
        "audit": _loss_weight_audit_manifest(stats),
    }


def _streamed_feature_manifest(
    feature_output: Path,
    stats: Mapping[str, object],
    parent_hashes: Mapping[str, str],
    output: Path,
    manifest_path: str | Path,
    scope: str,
) -> dict[str, object]:
    totals = stats["totals"]
    assert isinstance(totals, Counter)
    feature_parents = {
        **parent_hashes,
        "final_training_pairs": sha256_path(output),
        "final_training_pairs_manifest": sha256_path(manifest_path),
    }
    return {
        "artifact_type": "ranker_raw_pair_features",
        "artifact_version": RAW_FEATURE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "pair_kind": "training",
        "stage": "final_retrain",
        "row_layout": "one_row_per_training_pair",
        "feature_schema_version": RANKER_V2_SCHEMA_VERSION,
        "raw_feature_order": list(RAW_BASE_FEATURES),
        "row_count": stats["feature_count"],
        "counts": {
            "rows": stats["feature_count"],
            "label_0": int(totals["negative_count"]),
            "label_1": int(totals["positive_count"]),
        },
        "output_file": feature_output.name,
        "storage_format": "partitioned_parquet",
        "part_count": stats["feature_part_count"],
        "output_sha256": sha256_path(feature_output),
        "output_size_bytes": artifact_size_bytes(feature_output),
        "parent_hashes": feature_parents,
    }


def allowed_training_tracks(
    assignments: Mapping[str, str],
    stage: str,
) -> set[str]:
    if stage == "tuning":
        allowed_splits = {"set_a"}
    elif stage == "final_retrain":
        allowed_splits = {"set_a", "set_b", "remaining"}
    else:
        raise ValueError("training stage must be tuning or final_retrain")
    return {
        track_id for track_id, split in assignments.items() if split in allowed_splits
    }


def _pair_hash(query_id: str, candidate_id: str, source: str) -> str:
    return hashlib.sha256(
        f"{PAIR_SEED}\0{query_id}\0{candidate_id}\0{source}".encode("utf-8")
    ).hexdigest()


def _random_negatives(
    query_id: str,
    universe: Sequence[str],
    count: int,
    rejected: set[str],
    same_song: SameSong,
    is_positive: IsPositive,
    rejection_counts: Counter[str],
) -> list[str]:
    if count <= 0:
        return []
    selected: list[str] = []
    selected_set: set[str] = set()
    attempts = 0
    maximum_attempts = max(len(universe) * 4, count * 100)
    while len(selected) < count and attempts < maximum_attempts:
        digest = hashlib.sha256(
            f"{PAIR_SEED}\0{query_id}\0random\0{attempts}".encode("utf-8")
        ).digest()
        candidate_id = universe[int.from_bytes(digest[:8], "big") % len(universe)]
        attempts += 1
        if candidate_id == query_id:
            rejection_counts["query_self"] += 1
        elif candidate_id in rejected or candidate_id in selected_set:
            rejection_counts["duplicate_pair"] += 1
        elif same_song(query_id, candidate_id):
            rejection_counts["same_song"] += 1
        elif is_positive(query_id, candidate_id):
            rejection_counts["known_positive"] += 1
        else:
            selected.append(candidate_id)
            selected_set.add(candidate_id)
    return selected


def construct_query_pairs(
    query_id: str,
    positives: Mapping[str, frozenset[str]],
    candidates: Sequence[CandidateInput],
    allowed_tracks: set[str],
    random_universe: Sequence[str],
    same_song: SameSong,
    is_positive: IsPositive,
    is_positive_batch: IsPositiveBatch | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Construct one query's exact 1:3 curriculum and rejection audit."""
    if query_id not in allowed_tracks:
        raise ValueError(f"query endpoint is outside the training universe: {query_id}")
    selected_positives = {
        track_id: sources
        for track_id, sources in positives.items()
        if track_id in allowed_tracks
        and track_id != query_id
        and not same_song(query_id, track_id)
    }
    if not selected_positives:
        return [], {
            "positive_count": 0,
            "negative_count": 0,
            "candidate_aware_count": 0,
            "random_count": 0,
            "candidate_shortage": 0,
            "negative_shortage": 0,
            "rejections": {},
        }
    positive_ids = set(selected_positives)
    negative_target = NEGATIVE_RATIO * len(positive_ids)
    candidate_target = int(negative_target * CANDIDATE_AWARE_FRACTION)
    rejection_counts: Counter[str] = Counter()
    eligible_candidates: list[tuple[str, frozenset[str], Mapping[str, float]]] = []
    predicate_candidates: list[tuple[str, frozenset[str], Mapping[str, float]]] = []
    seen_candidates: set[str] = set()
    for candidate in candidates:
        if isinstance(candidate, Candidate):
            track_id = candidate.track_id
            recall_sources = candidate.sources
            recall_scores = candidate.recall_scores
        else:
            track_id = str(candidate["track_id"])
            recall_sources = frozenset(str(value) for value in candidate["recall_sources"])
            recall_scores = {
                str(name): float(value)
                for name, value in dict(candidate.get("recall_scores", {})).items()
            }
        if track_id == query_id:
            rejection_counts["query_self"] += 1
        elif track_id not in allowed_tracks:
            rejection_counts["outside_universe"] += 1
        elif track_id in seen_candidates:
            rejection_counts["duplicate_pair"] += 1
        elif same_song(query_id, track_id):
            rejection_counts["same_song"] += 1
        elif track_id in positive_ids:
            rejection_counts["known_positive"] += 1
        else:
            seen_candidates.add(track_id)
            predicate_candidates.append((track_id, recall_sources, recall_scores))
    predicate_results = (
        is_positive_batch(
            query_id,
            [track_id for track_id, _sources, _scores in predicate_candidates],
        )
        if is_positive_batch is not None
        else [
            is_positive(query_id, track_id)
            for track_id, _sources, _scores in predicate_candidates
        ]
    )
    if len(predicate_results) != len(predicate_candidates):
        raise ValueError("batch positive predicate returned the wrong number of results")
    for candidate, positive in zip(predicate_candidates, predicate_results, strict=True):
        if positive:
            rejection_counts["known_positive"] += 1
        else:
            eligible_candidates.append(candidate)
    eligible_candidates.sort(
        key=lambda item: _pair_hash(query_id, item[0], "candidate_aware")
    )
    candidate_selected = eligible_candidates[:candidate_target]
    rejected = positive_ids | {
        track_id for track_id, _sources, _scores in candidate_selected
    }
    random_target = negative_target - len(candidate_selected)
    random_selected = _random_negatives(
        query_id,
        random_universe,
        random_target,
        rejected,
        same_song,
        is_positive,
        rejection_counts,
    )
    if len(candidate_selected) + len(random_selected) != negative_target:
        raise ValueError(f"negative sampling shortage for query {query_id}")

    rows = [
        {
            "query_track_id": query_id,
            "candidate_track_id": track_id,
            "label": 1,
            "positive_sources": sorted(sources),
            "negative_source": None,
            "recall_sources": [],
        }
        for track_id, sources in sorted(selected_positives.items())
    ]
    rows.extend(
        {
            "query_track_id": query_id,
            "candidate_track_id": track_id,
            "label": 0,
            "positive_sources": [],
            "negative_source": "candidate_aware",
            "recall_sources": sorted(sources),
            "recall_scores": dict(scores),
        }
        for track_id, sources, scores in candidate_selected
    )
    rows.extend(
        {
            "query_track_id": query_id,
            "candidate_track_id": track_id,
            "label": 0,
            "positive_sources": [],
            "negative_source": "random",
            "recall_sources": [],
        }
        for track_id in random_selected
    )
    rows.sort(
        key=lambda row: (
            -int(row["label"]),
            _pair_hash(query_id, str(row["candidate_track_id"]), "output"),
        )
    )
    return rows, {
        "positive_count": len(positive_ids),
        "negative_count": negative_target,
        "candidate_aware_count": len(candidate_selected),
        "random_count": len(random_selected),
        "candidate_shortage": candidate_target - len(candidate_selected),
        "negative_shortage": 0,
        "rejections": dict(sorted(rejection_counts.items())),
    }


def iter_training_query_pairs(
    candidate_positive_records: Iterable[
        tuple[str, Sequence[CandidateInput], Mapping[str, frozenset[str]] | None]
    ],
    assignments: Mapping[str, str],
    *,
    stage: str,
    same_song: SameSong,
    is_positive: IsPositive,
    is_positive_batch: IsPositiveBatch | None = None,
) -> Iterator[tuple[str, list[dict[str, object]], dict[str, object]]]:
    """Construct bounded query groups directly from an in-memory recall stream."""
    allowed = allowed_training_tracks(assignments, stage)
    universe = tuple(sorted(allowed))
    previous_query: str | None = None
    for query_id, candidates, positives in candidate_positive_records:
        if previous_query is not None and query_id <= previous_query:
            raise ValueError("training query stream must be strictly sorted")
        previous_query = query_id
        if positives is None or query_id not in allowed:
            continue
        rows, audit = construct_query_pairs(
            query_id,
            positives,
            candidates,
            allowed,
            universe,
            same_song,
            is_positive,
            is_positive_batch,
        )
        if rows:
            yield query_id, rows, audit


def _write_streamed_rows(
    query_rows: Iterable[
        tuple[list[dict[str, object]], list[dict[str, object]], Mapping[str, object]]
    ],
    output: Path,
    feature_output: Path,
    rows_per_file: int,
) -> dict[str, object]:
    totals: Counter[str] = Counter()
    rejection_totals: Counter[str] = Counter()
    weak_source_totals: Counter[str] = Counter()
    recall_source_totals: Counter[str] = Counter()
    query_count = 0
    with PartitionedParquetWriter(
        output,
        training_pair_parquet_schema(),
        rows_per_file=rows_per_file,
    ) as pair_writer, PartitionedParquetWriter(
        feature_output,
        raw_feature_parquet_schema("training"),
        rows_per_file=rows_per_file,
    ) as feature_writer:
        for pairs, features, audit in query_rows:
            if len(pairs) != len(features):
                raise ValueError("streamed pair and feature query counts differ")
            query_count += 1
            for key in (
                "positive_count",
                "negative_count",
                "candidate_aware_count",
                "random_count",
                "candidate_shortage",
            ):
                totals[key] += int(audit[key])
            rejection_totals.update(audit["rejections"])
            for row in pairs:
                weak_source_totals.update(row["positive_sources"])
                recall_source_totals.update(row["recall_sources"])
            pair_writer.write_rows(
                {
                    "query_track_id": row["query_track_id"],
                    "candidate_track_id": row["candidate_track_id"],
                    "label": row["label"],
                    "positive_sources": row["positive_sources"],
                    "negative_source": row["negative_source"],
                    "recall_sources": row["recall_sources"],
                }
                for row in pairs
            )
            feature_writer.write_rows(features)
    return {
        "query_count": query_count,
        "pair_count": pair_writer.count,
        "feature_count": feature_writer.count,
        "pair_part_count": pair_writer.part_count,
        "feature_part_count": feature_writer.part_count,
        "totals": totals,
        "rejection_totals": rejection_totals,
        "weak_source_totals": weak_source_totals,
        "recall_source_totals": recall_source_totals,
    }


def iter_candidate_positives(
    candidate_pool_path: str | Path,
    positives: WeakPositiveSource,
) -> Iterator[tuple[dict[str, object], Mapping[str, frozenset[str]] | None]]:
    if isinstance(positives, Mapping):
        for record in iter_candidate_pool(candidate_pool_path):
            query_id = str(record["query_track_id"])
            yield record, positives.get(query_id)
        return

    positive_records = iter(iter_weak_positives(positives))
    current_positive = next(positive_records, None)
    previous_candidate_id: str | None = None
    previous_positive_id: str | None = None

    def advance_positive() -> tuple[str, dict[str, frozenset[str]]] | None:
        nonlocal previous_positive_id
        record = next(positive_records, None)
        if record is None:
            return None
        query_id = record[0]
        if previous_positive_id is not None and query_id <= previous_positive_id:
            raise ValueError("weak-positive queries must be strictly sorted")
        previous_positive_id = query_id
        return record

    if current_positive is not None:
        previous_positive_id = current_positive[0]
    for candidate_record in iter_candidate_pool(candidate_pool_path):
        candidate_query_id = str(candidate_record["query_track_id"])
        if (
            previous_candidate_id is not None
            and candidate_query_id <= previous_candidate_id
        ):
            raise ValueError("candidate-pool queries must be strictly sorted")
        previous_candidate_id = candidate_query_id
        while (
            current_positive is not None
            and current_positive[0] < candidate_query_id
        ):
            current_positive = advance_positive()
        if current_positive is None or current_positive[0] != candidate_query_id:
            yield candidate_record, None
            continue
        yield candidate_record, current_positive[1]
        current_positive = advance_positive()


def _streamed_pair_manifest(
    output: Path,
    stats: Mapping[str, object],
    parent_hashes: Mapping[str, str],
    scope: str,
) -> dict[str, object]:
    totals = stats["totals"]
    assert isinstance(totals, Counter)
    rejection_totals = stats["rejection_totals"]
    weak_source_totals = stats["weak_source_totals"]
    recall_source_totals = stats["recall_source_totals"]
    if totals["negative_count"] != NEGATIVE_RATIO * totals["positive_count"]:
        raise ValueError("streamed training artifact violates the 1:3 ratio")
    return {
        "artifact_type": "ranker_training_pairs",
        "artifact_version": TRAINING_PAIR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "stage": "final_retrain",
        "seed": PAIR_SEED,
        "negative_ratio": NEGATIVE_RATIO,
        "candidate_aware_target_fraction": CANDIDATE_AWARE_FRACTION,
        "query_count": stats["query_count"],
        "pair_count": stats["pair_count"],
        "counts": dict(sorted(totals.items())),
        "actual_candidate_aware_fraction": (
            totals["candidate_aware_count"] / totals["negative_count"]
        ),
        "rejection_counts": dict(sorted(rejection_totals.items())),
        "weak_positive_source_counts": dict(sorted(weak_source_totals.items())),
        "recall_source_totals": dict(sorted(recall_source_totals.items())),
        "candidate_pool_layout": "streamed_not_materialized",
        "pairs_file": output.name,
        "storage_format": "partitioned_parquet",
        "part_count": stats["pair_part_count"],
        "pairs_sha256": sha256_path(output),
        "pairs_size_bytes": artifact_size_bytes(output),
        "parent_hashes": parent_hashes,
    }


def write_training_pair_artifacts(
    candidate_pool_path: str | Path,
    positives: WeakPositiveSource,
    assignments: Mapping[str, str],
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    stage: str,
    same_song: SameSong,
    is_positive: IsPositive,
    parent_paths: Mapping[str, str | Path],
    scope: str,
    is_positive_batch: IsPositiveBatch | None = None,
) -> dict[str, object]:
    if scope not in {"formal", "smoke"}:
        raise ValueError("training-pair scope must be formal or smoke")
    allowed = allowed_training_tracks(assignments, stage)
    universe = tuple(sorted(allowed))
    totals: Counter[str] = Counter()
    rejection_totals: Counter[str] = Counter()
    query_count = 0

    def pair_rows() -> Iterator[dict[str, object]]:
        nonlocal query_count
        for record, query_positives in iter_candidate_positives(
            candidate_pool_path, positives
        ):
            query_id = str(record["query_track_id"])
            if query_positives is None or query_id not in allowed:
                continue
            rows, audit = construct_query_pairs(
                query_id,
                query_positives,
                record["candidates"],
                allowed,
                universe,
                same_song,
                is_positive,
                is_positive_batch,
            )
            if not rows:
                continue
            query_count += 1
            for key in (
                "positive_count",
                "negative_count",
                "candidate_aware_count",
                "random_count",
                "candidate_shortage",
            ):
                totals[key] += int(audit[key])
            rejection_totals.update(audit["rejections"])
            yield from rows

    output = Path(output_path)
    parquet_schema = None
    if output.suffix == ".parquet":
        parquet_schema = training_pair_parquet_schema()
    pair_count = write_row_artifact(pair_rows(), output, parquet_schema=parquet_schema)
    if query_count == 0 or pair_count == 0:
        raise ValueError("training-pair artifact has no eligible query pairs")
    if totals["negative_count"] != NEGATIVE_RATIO * totals["positive_count"]:
        raise ValueError("training-pair artifact violates the 1:3 ratio")
    manifest = {
        "artifact_type": "ranker_training_pairs",
        "artifact_version": TRAINING_PAIR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "stage": stage,
        "seed": PAIR_SEED,
        "negative_ratio": NEGATIVE_RATIO,
        "candidate_aware_target_fraction": CANDIDATE_AWARE_FRACTION,
        "query_count": query_count,
        "pair_count": pair_count,
        "counts": dict(sorted(totals.items())),
        "actual_candidate_aware_fraction": (
            totals["candidate_aware_count"] / totals["negative_count"]
        ),
        "rejection_counts": dict(sorted(rejection_totals.items())),
        "pairs_file": output.name,
        "storage_format": "parquet" if output.suffix == ".parquet" else "jsonl_gzip",
        "pairs_sha256": sha256_path(output),
        "pairs_size_bytes": artifact_size_bytes(output),
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def write_training_and_feature_artifacts(
    query_rows: Iterable[
        tuple[list[dict[str, object]], list[dict[str, object]], Mapping[str, object]]
    ],
    output_path: str | Path,
    manifest_path: str | Path,
    feature_output_path: str | Path,
    feature_manifest_path: str | Path,
    *,
    parent_paths: Mapping[str, str | Path],
    scope: str,
    rows_per_file: int = 250_000,
) -> tuple[dict[str, object], dict[str, object]]:
    """Write streamed final-retrain pairs and features in one partitioned pass."""
    if scope not in {"formal", "smoke"}:
        raise ValueError("training scope must be formal or smoke")
    output = Path(output_path)
    feature_output = Path(feature_output_path)
    stats = _write_streamed_rows(query_rows, output, feature_output, rows_per_file)
    query_count = int(stats["query_count"])
    pair_count = int(stats["pair_count"])
    feature_count = int(stats["feature_count"])
    if query_count == 0 or pair_count == 0 or pair_count != feature_count:
        raise ValueError("streamed training pair and feature artifacts are inconsistent")
    parent_hashes = {
        name: sha256_path(path) for name, path in sorted(parent_paths.items())
    }
    manifest = _streamed_pair_manifest(output, stats, parent_hashes, scope)
    write_json_atomic(manifest, manifest_path)
    feature_manifest = _streamed_feature_manifest(
        feature_output, stats, parent_hashes, output, manifest_path, scope
    )
    write_json_atomic(feature_manifest, feature_manifest_path)
    return manifest, feature_manifest


def load_training_pair_manifest(
    manifest_path: str | Path,
    pairs_path: str | Path,
    *,
    expected_scope: str,
    expected_stage: str,
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != "ranker_training_pairs":
        raise ValueError("training-pair artifact type mismatch")
    if manifest.get("artifact_version") != TRAINING_PAIR_VERSION:
        raise ValueError("training-pair artifact version mismatch")
    if manifest.get("scope") != expected_scope:
        raise ValueError("training-pair scope mismatch")
    if manifest.get("stage") != expected_stage:
        raise ValueError("training-pair stage mismatch")
    pairs = Path(pairs_path)
    if manifest.get("pairs_file") != pairs.name:
        raise ValueError("training-pair output path mismatch")
    if manifest.get("pairs_sha256") != sha256_path(pairs):
        raise ValueError("training-pair output hash mismatch")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("training-pair counts are missing")
    if int(counts.get("negative_count", -1)) != NEGATIVE_RATIO * int(
        counts.get("positive_count", -1)
    ):
        raise ValueError("training-pair manifest violates the 1:3 ratio")
    return manifest
