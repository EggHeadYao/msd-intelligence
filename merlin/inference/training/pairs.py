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
from ..recall.streaming import EncodedCandidates, SOURCE_NAMES
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


TRAINING_PAIR_VERSION = "merlin_training_pairs_v6"
PAIR_SEED = 42
NEGATIVE_RATIO = 3
CANDIDATE_AWARE_FRACTION = 0.75
LOSS_WEIGHT_STRATEGY = "positive_modality_balance_and_candidate_curriculum"
MAX_CANDIDATE_SAMPLE_WEIGHT = 20.0
AUDIO_POSITIVE_SOURCE = "audio_derived"
RELATION_POSITIVE_SOURCES = frozenset(("same_artist", "tag_derived"))
IsPositive = Callable[[str, str], bool]
IsPositiveBatch = Callable[[str, Sequence[str]], Sequence[bool]]
IsPositiveEncodedBatch = Callable[
    [EncodedCandidates, np.ndarray],
    Sequence[bool] | np.ndarray,
]
IsPositivePairs = Callable[[Sequence[tuple[str, str]]], Sequence[bool]]
SameSong = Callable[[str, str], bool]
WeakPositiveMap = Mapping[str, Mapping[str, frozenset[str]]]
WeakPositiveSource = str | Path | WeakPositiveMap
CandidateInput = Mapping[str, object] | Candidate
CandidateCollection = Sequence[CandidateInput] | EncodedCandidates
CANDIDATE_SAMPLING_STRATEGY = "source_balanced_round_robin_v1"


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


def _positive_loss_weights(
    positives: Mapping[str, frozenset[str]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Give audio and relation evidence equal loss mass within each query."""
    count = len(positives)
    if count == 0:
        return {}, {"positive_audio": 0.0, "positive_relation": 0.0}
    audio = {
        track_id for track_id, sources in positives.items()
        if AUDIO_POSITIVE_SOURCE in sources
    }
    relation = {
        track_id for track_id, sources in positives.items()
        if RELATION_POSITIVE_SOURCES.intersection(sources)
    }
    unknown = set(positives).difference(audio | relation)
    if unknown:
        raise ValueError("positive source is outside the modality-weight contract")
    if audio and relation:
        audio_mass = relation_mass = count / 2.0
    elif audio:
        audio_mass, relation_mass = float(count), 0.0
    else:
        audio_mass, relation_mass = 0.0, float(count)
    weights = {
        track_id: (
            (audio_mass / len(audio) if track_id in audio else 0.0)
            + (relation_mass / len(relation) if track_id in relation else 0.0)
        )
        for track_id in positives
    }
    if abs(sum(weights.values()) - count) > 1e-9:
        raise AssertionError("positive modality weights changed total loss mass")
    return weights, {
        "positive_audio": audio_mass,
        "positive_relation": relation_mass,
    }


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
        "positive_modality_balance": "equal_mass_per_query_when_both_exist",
        "positive_audio_weight_sum": float(totals.get("positive_audio", 0.0)),
        "positive_relation_weight_sum": float(
            totals.get("positive_relation", 0.0)
        ),
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
    candidate_aware_fraction: float,
) -> dict[str, object]:
    totals = stats["totals"]
    assert isinstance(totals, Counter)
    loss_weight_totals = stats["loss_weight_totals"]
    effective_pair_count = int(stats.get("effective_pair_count", stats["feature_count"]))
    stored_counts = stats.get("stored_counts", totals)
    feature_parents = {
        **parent_hashes,
        "training_pairs": sha256_path(output),
        "training_pairs_manifest": sha256_path(manifest_path),
    }
    return {
        "artifact_type": "ranker_raw_pair_features",
        "artifact_version": RAW_FEATURE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "pair_kind": "training",
        "stage": "final_retrain",
        "row_layout": (
            "random_replacement_delta"
            if effective_pair_count != int(stats["feature_count"])
            else "one_row_per_training_pair"
        ),
        "feature_schema_version": FEATURE_SCHEMA,
        "raw_feature_order": list(RAW_BASE_FEATURES),
        "row_count": stats["feature_count"],
        "effective_row_count": effective_pair_count,
        "counts": {
            "rows": stats["feature_count"],
            "label_0": int(stored_counts["negative_count"]),
            "label_1": int(stored_counts["positive_count"]),
        },
        "loss_weighting": _loss_weight_manifest(
            loss_weight_totals,
            candidate_aware_fraction,
            stats,
        ),
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
    allowed_splits = training_splits(stage)
    return {
        track_id for track_id, split in assignments.items() if split in allowed_splits
    }


def training_splits(stage: str) -> frozenset[str]:
    if stage == "tuning":
        return frozenset(("set_a",))
    if stage == "final_retrain":
        return frozenset(("set_a", "set_b", "set_c", "remaining"))
    raise ValueError("training stage must be tuning or final_retrain")


def _pair_hash(query_id: str, candidate_id: str, source: str) -> str:
    return hashlib.sha256(
        f"{PAIR_SEED}\0{query_id}\0{candidate_id}\0{source}".encode("utf-8")
    ).hexdigest()


_UINT64_MASK = (1 << 64) - 1


def _random_query_seed(query_id: str) -> int:
    digest = hashlib.sha256(
        f"{PAIR_SEED}\0{query_id}\0random".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _random_universe_index(seed: int, attempt: int, size: int) -> int:
    """Map a query-local counter to a stable pseudo-random universe index."""
    value = (seed + attempt + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) % size


def _random_negatives(
    query_id: str,
    universe: Sequence[str],
    count: int,
    rejected: set[str],
    same_song: SameSong,
    is_positive: IsPositive,
    is_positive_batch: IsPositiveBatch | None,
    rejection_counts: Counter[str],
) -> list[str]:
    if count <= 0:
        return []
    selected: list[str] = []
    selected_set: set[str] = set()
    attempts = 0
    maximum_attempts = max(len(universe) * 4, count * 100)
    random_seed = _random_query_seed(query_id)
    while len(selected) < count and attempts < maximum_attempts:
        batch_end = min(attempts + max(64, count * 4), maximum_attempts)
        proposals = []
        for attempt in range(attempts, batch_end):
            proposals.append(
                universe[_random_universe_index(random_seed, attempt, len(universe))]
            )
        predicate_ids = list(dict.fromkeys(
            candidate_id
            for candidate_id in proposals
            if candidate_id != query_id
            and candidate_id not in rejected
            and not same_song(query_id, candidate_id)
        ))
        predicate_results = (
            is_positive_batch(query_id, predicate_ids)
            if is_positive_batch is not None
            else [is_positive(query_id, candidate_id) for candidate_id in predicate_ids]
        )
        if len(predicate_results) != len(predicate_ids):
            raise ValueError("batch positive predicate returned the wrong number of results")
        positive_by_id = dict(zip(predicate_ids, predicate_results, strict=True))
        for candidate_id in proposals:
            attempts += 1
            if candidate_id == query_id:
                rejection_counts["query_self"] += 1
            elif candidate_id in rejected or candidate_id in selected_set:
                rejection_counts["duplicate_pair"] += 1
            elif same_song(query_id, candidate_id):
                rejection_counts["same_song"] += 1
            elif positive_by_id[candidate_id]:
                rejection_counts["known_positive"] += 1
            else:
                selected.append(candidate_id)
                selected_set.add(candidate_id)
                if len(selected) == count:
                    break
    return selected


def sample_random_negatives(
    query_id: str,
    universe: Sequence[str],
    count: int,
    rejected: set[str],
    same_song: SameSong,
    is_positive: IsPositive,
    is_positive_batch: IsPositiveBatch | None,
) -> tuple[list[str], dict[str, int]]:
    """Sample a deterministic random-only replacement set with an audit."""
    rejection_counts: Counter[str] = Counter()
    selected = _random_negatives(
        query_id,
        universe,
        count,
        rejected,
        same_song,
        is_positive,
        is_positive_batch,
        rejection_counts,
    )
    if len(selected) != count:
        raise ValueError(f"random-negative shortage for query {query_id}")
    return selected, dict(sorted(rejection_counts.items()))


def sample_random_negatives_many(
    requests: Sequence[tuple[str, int, set[str]]],
    universe: Sequence[str],
    same_song: SameSong,
    is_positive_pairs: IsPositivePairs,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Sample many queries while batching their positive-label checks."""
    selected, by_query = sample_random_negatives_many_by_query(
        requests,
        universe,
        same_song,
        is_positive_pairs,
    )
    totals: Counter[str] = Counter()
    for counts in by_query.values():
        totals.update(counts)
    return selected, dict(sorted(totals.items()))


def sample_random_negatives_many_by_query(
    requests: Sequence[tuple[str, int, set[str]]],
    universe: Sequence[str],
    same_song: SameSong,
    is_positive_pairs: IsPositivePairs,
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """Sample many queries and retain each query's rejection audit."""
    if not universe:
        raise ValueError("random-negative universe is empty")
    states: dict[str, dict[str, object]] = {}
    for query_id, count, rejected in requests:
        if query_id in states:
            raise ValueError(f"duplicate random-negative request: {query_id}")
        if count < 0:
            raise ValueError("random-negative request count must be non-negative")
        states[query_id] = {
            "target": count,
            "rejected": rejected,
            "selected": [],
            "selected_set": set(),
            "attempts": 0,
            "maximum_attempts": max(len(universe) * 4, count * 100),
            "random_seed": _random_query_seed(query_id),
        }
    rejection_counts: dict[str, Counter[str]] = {
        query_id: Counter() for query_id in states
    }
    unfinished = [query_id for query_id, state in states.items() if state["target"]]
    while unfinished:
        proposals_by_query: dict[str, list[str]] = {}
        predicate_pairs: list[tuple[str, str]] = []
        for query_id in unfinished:
            state = states[query_id]
            attempts = int(state["attempts"])
            target = int(state["target"])
            maximum = int(state["maximum_attempts"])
            batch_end = min(attempts + max(64, target * 4), maximum)
            proposals = []
            for attempt in range(attempts, batch_end):
                proposals.append(
                    universe[_random_universe_index(
                        int(state["random_seed"]),
                        attempt,
                        len(universe),
                    )]
                )
            proposals_by_query[query_id] = proposals
            rejected = state["rejected"]
            selected_set = state["selected_set"]
            assert isinstance(rejected, set) and isinstance(selected_set, set)
            predicate_pairs.extend(
                (query_id, candidate_id)
                for candidate_id in proposals
                if candidate_id != query_id
                and candidate_id not in rejected
                and candidate_id not in selected_set
                and not same_song(query_id, candidate_id)
            )
        predicate_pairs = list(dict.fromkeys(predicate_pairs))
        predicate_results = is_positive_pairs(predicate_pairs)
        if len(predicate_results) != len(predicate_pairs):
            raise ValueError("pair positive predicate returned the wrong number of results")
        positive_by_pair = dict(zip(predicate_pairs, predicate_results, strict=True))
        next_unfinished = []
        for query_id in unfinished:
            state = states[query_id]
            rejected = state["rejected"]
            selected = state["selected"]
            selected_set = state["selected_set"]
            assert isinstance(rejected, set)
            assert isinstance(selected, list) and isinstance(selected_set, set)
            for candidate_id in proposals_by_query[query_id]:
                state["attempts"] = int(state["attempts"]) + 1
                if candidate_id == query_id:
                    rejection_counts[query_id]["query_self"] += 1
                elif candidate_id in rejected or candidate_id in selected_set:
                    rejection_counts[query_id]["duplicate_pair"] += 1
                elif same_song(query_id, candidate_id):
                    rejection_counts[query_id]["same_song"] += 1
                elif positive_by_pair[(query_id, candidate_id)]:
                    rejection_counts[query_id]["known_positive"] += 1
                else:
                    selected.append(candidate_id)
                    selected_set.add(candidate_id)
                    if len(selected) == int(state["target"]):
                        break
            if len(selected) < int(state["target"]):
                if int(state["attempts"]) >= int(state["maximum_attempts"]):
                    raise ValueError(f"random-negative shortage for query {query_id}")
                next_unfinished.append(query_id)
        unfinished = next_unfinished
    return (
        {query_id: list(state["selected"]) for query_id, state in states.items()},
        {
            query_id: dict(sorted(counts.items()))
            for query_id, counts in rejection_counts.items()
        },
    )


@dataclass(slots=True)
class PreparedQueryPairs:
    query_id: str
    selected_positives: Mapping[str, frozenset[str]]
    candidates: CandidateCollection
    candidate_selected: list[tuple[str, object]]
    negative_target: int
    candidate_target: int
    unrecalled_positive_count: int
    rejection_counts: Counter[str]


def _candidate_sources(
    candidates: CandidateCollection,
    recall_evidence: object,
) -> frozenset[str]:
    if isinstance(candidates, EncodedCandidates):
        mask = int(candidates.source_masks[int(recall_evidence)])
        return frozenset(
            source
            for index, source in enumerate(SOURCE_NAMES)
            if mask & (1 << index)
        )
    sources, _scores = recall_evidence  # type: ignore[misc]
    return frozenset(str(source) for source in sources)


def _source_balanced_candidate_sample(
    query_id: str,
    eligible: Sequence[tuple[str, object]],
    candidates: CandidateCollection,
    target: int,
) -> list[tuple[str, object]]:
    """Sample hard negatives evenly across their observable recall sources."""
    if target <= 0 or not eligible:
        return []
    buckets: dict[str, list[tuple[str, object]]] = {}
    if isinstance(candidates, EncodedCandidates):
        positions = np.fromiter(
            (int(evidence) for _track_id, evidence in eligible),
            dtype=np.int64,
            count=len(eligible),
        )
        masks = candidates.source_masks[positions]
        source_bits = np.asarray(
            tuple(1 << index for index in range(len(SOURCE_NAMES))),
            dtype=masks.dtype,
        )
        memberships = (masks[:, None] & source_bits[None, :]) != 0
        frequencies = np.count_nonzero(memberships, axis=0)
        tie_order = sorted(
            range(len(SOURCE_NAMES)),
            key=lambda index: _pair_hash(
                query_id, SOURCE_NAMES[index], "candidate_source_owner"
            ),
        )
        priorities = np.empty(len(SOURCE_NAMES), dtype=np.int64)
        priorities[tie_order] = np.arange(len(SOURCE_NAMES), dtype=np.int64)
        costs = frequencies * len(SOURCE_NAMES) + priorities
        owners = np.argmin(
            np.where(memberships, costs[None, :], np.iinfo(np.int64).max),
            axis=1,
        )
        for item, owner, mask in zip(eligible, owners, masks, strict=True):
            source = SOURCE_NAMES[int(owner)] if int(mask) else "unknown"
            buckets.setdefault(source, []).append(item)
    else:
        sources_by_candidate = [
            _candidate_sources(candidates, evidence) or frozenset(("unknown",))
            for _track_id, evidence in eligible
        ]
        source_frequency = Counter(
            source for sources in sources_by_candidate for source in sources
        )
        for item, sources in zip(eligible, sources_by_candidate, strict=True):
            track_id = item[0]
            owner = min(
                sources,
                key=lambda source: (
                    source_frequency[source],
                    _pair_hash(query_id, track_id, f"candidate_source:{source}"),
                    source,
                ),
            )
            buckets.setdefault(owner, []).append(item)
    source_order = sorted(
        buckets,
        key=lambda source: (
            _pair_hash(query_id, source, "candidate_source_order"),
            source,
        ),
    )
    ordered = {
        source: nsmallest(
            target,
            items,
            key=lambda item: _pair_hash(
                query_id, item[0], f"candidate_aware:{source}"
            ),
        )
        for source, items in buckets.items()
    }
    positions = dict.fromkeys(source_order, 0)
    selected: list[tuple[str, object]] = []
    while len(selected) < target:
        advanced = False
        for source in source_order:
            position = positions[source]
            if position >= len(ordered[source]):
                continue
            selected.append(ordered[source][position])
            positions[source] = position + 1
            advanced = True
            if len(selected) == target:
                break
        if not advanced:
            break
    return selected


def _empty_pair_audit() -> dict[str, object]:
    return {
        "positive_count": 0,
        "unrecalled_positive_count": 0,
        "negative_count": 0,
        "candidate_aware_count": 0,
        "random_count": 0,
        "candidate_shortage": 0,
        "negative_shortage": 0,
        "loss_weight_sums": {
            "positive": 0.0,
            "candidate_aware": 0.0,
            "random": 0.0,
        },
        "loss_weight_audit": {
            "candidate_weight": 0.0,
            "candidate_count": 0,
            "effective_candidate_aware_fraction": 0.0,
        },
        "rejections": {},
    }


def prepare_query_pairs(
    query_id: str,
    positives: Mapping[str, frozenset[str]],
    candidates: CandidateCollection,
    allowed_tracks: set[str],
    same_song: SameSong,
    is_positive: IsPositive,
    is_positive_batch: IsPositiveBatch | None = None,
    candidate_aware_fraction: float = CANDIDATE_AWARE_FRACTION,
    is_positive_encoded_batch: IsPositiveEncodedBatch | None = None,
) -> PreparedQueryPairs | None:
    """Select positives and candidate-aware negatives before random backfill."""
    if not 0.0 <= candidate_aware_fraction <= 1.0:
        raise ValueError("candidate-aware fraction must be in [0, 1]")
    if query_id not in allowed_tracks:
        raise ValueError(f"query endpoint is outside the training universe: {query_id}")
    allowed_positives = {
        track_id: sources
        for track_id, sources in positives.items()
        if track_id in allowed_tracks
        and track_id != query_id
        and not same_song(query_id, track_id)
    }
    if isinstance(candidates, EncodedCandidates):
        positive_items = tuple(allowed_positives.items())
        positive_codes = np.fromiter(
            (
                candidates.codec.code(track_id)
                for track_id, _sources in positive_items
            ),
            dtype=np.int32,
            count=len(positive_items),
        )
        recalled = np.isin(positive_codes, candidates.codes, assume_unique=True)
        selected_positives = dict(
            item
            for item, present in zip(positive_items, recalled, strict=True)
            if present
        )
    else:
        candidate_ids = {
            (
                candidate.track_id
                if isinstance(candidate, Candidate)
                else str(candidate["track_id"])
            )
            for candidate in candidates
        }
        selected_positives = {
            track_id: sources
            for track_id, sources in allowed_positives.items()
            if track_id in candidate_ids
        }
    if not selected_positives:
        return None
    positive_ids = set(selected_positives)
    negative_target = NEGATIVE_RATIO * len(positive_ids)
    candidate_target = int(negative_target * candidate_aware_fraction)
    rejection_counts: Counter[str] = Counter()
    eligible_candidates: list[tuple[str, object]] = []
    predicate_candidates: list[tuple[str, object]] = []
    seen_candidates: set[str] = set()

    def records() -> Iterator[tuple[str, object]]:
        if isinstance(candidates, EncodedCandidates):
            for position in range(len(candidates)):
                yield candidates.track_id(position), position
            return
        for candidate in candidates:
            if isinstance(candidate, Candidate):
                evidence = (candidate.sources, candidate.recall_scores)
                yield candidate.track_id, evidence
            else:
                sources = frozenset(str(value) for value in candidate["recall_sources"])
                scores = {
                    str(name): float(value)
                    for name, value in dict(candidate.get("recall_scores", {})).items()
                }
                yield str(candidate["track_id"]), (sources, scores)

    if candidate_target == 0:
        pass
    elif isinstance(candidates, EncodedCandidates):
        codes = candidates.codes
        remaining = np.ones(len(codes), dtype=np.bool_)
        query_code = candidates.codec.code(query_id)
        query_self = remaining & (codes == query_code)
        rejection_counts["query_self"] += int(np.count_nonzero(query_self))
        remaining &= ~query_self
        outside = remaining & ~candidates.codec.allowed[codes]
        rejection_counts["outside_universe"] += int(np.count_nonzero(outside))
        remaining &= ~outside
        same_song_mask = remaining & candidates.codec.same_song_mask(query_code, codes)
        rejection_counts["same_song"] += int(np.count_nonzero(same_song_mask))
        remaining &= ~same_song_mask
        same_artist = remaining & candidates.codec.same_artist_mask(query_code, codes)
        rejection_counts["known_positive"] += int(np.count_nonzero(same_artist))
        remaining &= ~same_artist
        positive_codes = np.asarray(
            [candidates.codec.code(track_id) for track_id in positive_ids],
            dtype=np.int32,
        )
        known_positive = remaining & np.isin(codes, positive_codes)
        rejection_counts["known_positive"] += int(np.count_nonzero(known_positive))
        remaining &= ~known_positive
        remaining_positions = np.flatnonzero(remaining)
        if is_positive_encoded_batch is None:
            predicate_candidates.extend(
                (candidates.track_id(int(position)), int(position))
                for position in remaining_positions
            )
        else:
            encoded_results = np.asarray(
                is_positive_encoded_batch(candidates, remaining_positions),
                dtype=np.bool_,
            )
            if encoded_results.shape != remaining_positions.shape:
                raise ValueError(
                    "encoded positive predicate returned the wrong result shape"
                )
            rejection_counts["known_positive"] += int(
                np.count_nonzero(encoded_results)
            )
            eligible_candidates.extend(
                (candidates.track_id(int(position)), int(position))
                for position in remaining_positions[~encoded_results]
            )
    else:
        for track_id, recall_evidence in records():
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
                predicate_candidates.append((track_id, recall_evidence))
    predicate_results = [] if not predicate_candidates else (
        is_positive_batch(
            query_id,
            [track_id for track_id, _evidence in predicate_candidates],
        )
        if is_positive_batch is not None
        else [
            is_positive(query_id, track_id)
            for track_id, _evidence in predicate_candidates
        ]
    )
    if len(predicate_results) != len(predicate_candidates):
        raise ValueError("batch positive predicate returned the wrong number of results")
    for candidate, positive in zip(predicate_candidates, predicate_results, strict=True):
        if positive:
            rejection_counts["known_positive"] += 1
        else:
            eligible_candidates.append(candidate)
    candidate_selected = _source_balanced_candidate_sample(
        query_id,
        eligible_candidates,
        candidates,
        candidate_target,
    )
    return PreparedQueryPairs(
        query_id,
        selected_positives,
        candidates,
        candidate_selected,
        negative_target,
        candidate_target,
        len(allowed_positives) - len(selected_positives),
        rejection_counts,
    )


def finish_query_pairs(
    prepared: PreparedQueryPairs,
    random_selected: Sequence[str],
    random_rejections: Mapping[str, int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Finish one prepared query with its deterministic random backfill."""
    query_id = prepared.query_id
    candidate_selected = prepared.candidate_selected
    negative_target = prepared.negative_target
    candidate_target = prepared.candidate_target
    selected_positives = prepared.selected_positives
    candidates = prepared.candidates
    rejection_counts = prepared.rejection_counts.copy()
    rejection_counts.update(random_rejections)
    random_selected = list(random_selected)
    if len(candidate_selected) + len(random_selected) != negative_target:
        raise ValueError(f"negative sampling shortage for query {query_id}")

    def evidence(value: object) -> tuple[frozenset[str], Mapping[str, float]]:
        if isinstance(candidates, EncodedCandidates):
            return candidates.evidence(int(value))
        sources, scores = value  # type: ignore[misc]
        return sources, scores

    candidate_weight, random_weight = _negative_loss_weights(
        negative_target,
        len(candidate_selected),
        len(random_selected),
        candidate_target,
    )
    positive_ids = set(selected_positives)
    positive_weights, positive_weight_sums = _positive_loss_weights(
        selected_positives
    )
    rows = [
        {
            "query_track_id": query_id,
            "candidate_track_id": track_id,
            "label": 1,
            SAMPLE_WEIGHT_COLUMN: positive_weights[track_id],
            "positive_sources": sorted(sources),
            "negative_source": None,
            "recall_sources": [],
        }
        for track_id, sources in sorted(selected_positives.items())
    ]
    for track_id, recall_evidence in candidate_selected:
        recall_sources, recall_scores = evidence(recall_evidence)
        rows.append({
            "query_track_id": query_id,
            "candidate_track_id": track_id,
            "label": 0,
            SAMPLE_WEIGHT_COLUMN: candidate_weight,
            "positive_sources": [],
            "negative_source": "candidate_aware",
            "recall_sources": sorted(recall_sources),
            "recall_scores": dict(recall_scores),
        })
    rows.extend(
        {
            "query_track_id": query_id,
            "candidate_track_id": track_id,
            "label": 0,
            SAMPLE_WEIGHT_COLUMN: random_weight,
            "positive_sources": [],
            "negative_source": "random",
            "recall_sources": [],
        }
        for track_id in random_selected
    )
    return rows, {
        "positive_count": len(positive_ids),
        "unrecalled_positive_count": prepared.unrecalled_positive_count,
        "negative_count": negative_target,
        "candidate_aware_count": len(candidate_selected),
        "random_count": len(random_selected),
        "candidate_shortage": candidate_target - len(candidate_selected),
        "negative_shortage": 0,
        "loss_weight_sums": {
            "positive": float(len(positive_ids)),
            **positive_weight_sums,
            "candidate_aware": candidate_weight * len(candidate_selected),
            "random": random_weight * len(random_selected),
        },
        "loss_weight_audit": {
            "candidate_weight": candidate_weight,
            "candidate_count": len(candidate_selected),
            "effective_candidate_aware_fraction": (
                candidate_weight * len(candidate_selected) / negative_target
            ),
        },
        "rejections": dict(sorted(rejection_counts.items())),
    }


def construct_query_pairs(
    query_id: str,
    positives: Mapping[str, frozenset[str]],
    candidates: CandidateCollection,
    allowed_tracks: set[str],
    random_universe: Sequence[str],
    same_song: SameSong,
    is_positive: IsPositive,
    is_positive_batch: IsPositiveBatch | None = None,
    candidate_aware_fraction: float = CANDIDATE_AWARE_FRACTION,
    is_positive_encoded_batch: IsPositiveEncodedBatch | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Construct one query's exact 1:3 curriculum and rejection audit."""
    prepared = prepare_query_pairs(
        query_id,
        positives,
        candidates,
        allowed_tracks,
        same_song,
        is_positive,
        is_positive_batch,
        candidate_aware_fraction,
        is_positive_encoded_batch,
    )
    if prepared is None:
        return [], _empty_pair_audit()
    random_target = prepared.negative_target - len(prepared.candidate_selected)
    rejected = set(prepared.selected_positives) | {
        track_id for track_id, _evidence in prepared.candidate_selected
    }
    rejection_counts: Counter[str] = Counter()
    random_selected = _random_negatives(
        query_id,
        random_universe,
        random_target,
        rejected,
        same_song,
        is_positive,
        is_positive_batch,
        rejection_counts,
    )
    return finish_query_pairs(prepared, random_selected, rejection_counts)


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
        | StreamCheckpoint
        | StreamTableBatch
    ],
    output: Path,
    feature_output: Path,
    rows_per_file: int,
    *,
    candidate_aware_fraction: float,
    checkpoint_path: Path | None = None,
    checkpoint_contract: Mapping[str, object] | None = None,
    initial_checkpoint: Mapping[str, object] | None = None,
) -> dict[str, object]:
    initial = initial_checkpoint or {}
    totals: Counter[str] = Counter(initial.get("totals", {}))
    loss_weight_totals: Counter[str] = Counter(
        initial.get("loss_weight_totals", {})
    )
    loss_weight_shape_histogram: Counter[str] = Counter(
        initial.get("loss_weight_shape_histogram", {})
    )
    legacy_candidate_weights = Counter(
        initial.get("legacy_candidate_weight_histogram", {})
    )
    legacy_effective_fractions = Counter(
        initial.get("legacy_effective_fraction_histogram", {})
    )
    legacy_queries_without_candidate = int(
        initial.get("legacy_queries_without_candidate_aware", 0)
    )
    rejection_totals: Counter[str] = Counter(initial.get("rejection_totals", {}))
    weak_source_totals: Counter[str] = Counter(initial.get("weak_source_totals", {}))
    recall_source_totals: Counter[str] = Counter(initial.get("recall_source_totals", {}))
    query_count = int(initial.get("query_count", 0))
    resume = initial_checkpoint is not None
    with PartitionedParquetWriter(
        output,
        training_pair_parquet_schema(),
        rows_per_file=rows_per_file,
        resume=resume,
    ) as pair_writer, PartitionedParquetWriter(
        feature_output,
        raw_feature_parquet_schema("training"),
        rows_per_file=rows_per_file,
        resume=resume,
    ) as feature_writer:
        expected_pairs = int(initial.get("pair_count", 0))
        expected_features = int(initial.get("feature_count", 0))
        if pair_writer.count != expected_pairs or feature_writer.count != expected_features:
            raise ValueError("checkpoint row counts do not match recoverable Parquet parts")
        if checkpoint_path is not None and initial_checkpoint is None:
            write_json_atomic({
                "artifact_type": "final_retrain_checkpoint",
                "artifact_version": 3,
                "contract": dict(checkpoint_contract or {}),
                "processed_queries": 0,
                "total_queries": int((checkpoint_contract or {}).get("query_count", 0)),
                "query_count": 0,
                "pair_count": 0,
                "feature_count": 0,
                "pair_part_count": 0,
                "feature_part_count": 0,
                "totals": {},
                "loss_weight_totals": {},
                "loss_weight_shape_histogram": {},
                "rejection_totals": {},
                "weak_source_totals": {},
                "recall_source_totals": {},
            }, checkpoint_path)
        for item in query_rows:
            if isinstance(item, StreamCheckpoint):
                pair_writer.checkpoint()
                feature_writer.checkpoint()
                if checkpoint_path is not None:
                    write_json_atomic({
                        "artifact_type": "final_retrain_checkpoint",
                        "artifact_version": 3,
                        "contract": dict(checkpoint_contract or {}),
                        "processed_queries": item.processed_queries,
                        "total_queries": item.total_queries,
                        "query_count": query_count,
                        "pair_count": pair_writer.count,
                        "feature_count": feature_writer.count,
                        "pair_part_count": pair_writer.part_count,
                        "feature_part_count": feature_writer.part_count,
                        "totals": dict(totals),
                        "loss_weight_totals": dict(loss_weight_totals),
                        "loss_weight_shape_histogram": dict(
                            loss_weight_shape_histogram
                        ),
                        **({
                            "legacy_candidate_weight_histogram": dict(
                                legacy_candidate_weights
                            ),
                            "legacy_effective_fraction_histogram": dict(
                                legacy_effective_fractions
                            ),
                            "legacy_queries_without_candidate_aware": (
                                legacy_queries_without_candidate
                            ),
                        } if legacy_candidate_weights or legacy_effective_fractions else {}),
                        "rejection_totals": dict(rejection_totals),
                        "weak_source_totals": dict(weak_source_totals),
                        "recall_source_totals": dict(recall_source_totals),
                    }, checkpoint_path)
                continue
            if isinstance(item, StreamTableBatch):
                if item.pairs.num_rows != item.features.num_rows:
                    raise ValueError("streamed pair and feature table counts differ")
                for audit in item.audits:
                    query_count += 1
                    for key in (
                        "positive_count",
                        "unrecalled_positive_count",
                        "negative_count",
                        "candidate_aware_count",
                        "random_count",
                        "candidate_shortage",
                    ):
                        totals[key] += int(audit[key])
                    loss_weight_totals.update(audit["loss_weight_sums"])
                    loss_weight_shape_histogram[_loss_weight_shape_key(audit)] += 1
                    rejection_totals.update(audit["rejections"])
                weak_source_totals.update(item.weak_source_totals)
                recall_source_totals.update(item.recall_source_totals)
                pair_writer.write_table(item.pairs)
                feature_writer.write_table(item.features)
                continue
            pairs, features, audit = item
            if len(pairs) != len(features):
                raise ValueError("streamed pair and feature query counts differ")
            query_count += 1
            for key in (
                "positive_count",
                "unrecalled_positive_count",
                "negative_count",
                "candidate_aware_count",
                "random_count",
                "candidate_shortage",
            ):
                totals[key] += int(audit[key])
            loss_weight_totals.update(audit["loss_weight_sums"])
            loss_weight_shape_histogram[_loss_weight_shape_key(audit)] += 1
            rejection_totals.update(audit["rejections"])
            for row in pairs:
                weak_source_totals.update(row["positive_sources"])
                recall_source_totals.update(row["recall_sources"])
            pair_writer.write_rows(
                {
                    "query_track_id": row["query_track_id"],
                    "candidate_track_id": row["candidate_track_id"],
                    "label": row["label"],
                    SAMPLE_WEIGHT_COLUMN: row[SAMPLE_WEIGHT_COLUMN],
                    "positive_sources": row["positive_sources"],
                    "negative_source": row["negative_source"],
                    "recall_sources": row["recall_sources"],
                }
                for row in pairs
            )
            feature_writer.write_rows(
                {
                    **feature,
                    "negative_source": pair["negative_source"],
                }
                for pair, feature in zip(pairs, features, strict=True)
            )
    return {
        "query_count": query_count,
        "pair_count": pair_writer.count,
        "feature_count": feature_writer.count,
        "pair_part_count": pair_writer.part_count,
        "feature_part_count": feature_writer.part_count,
        "totals": totals,
        "loss_weight_totals": loss_weight_totals,
        "loss_weight_shape_histogram": loss_weight_shape_histogram,
        "legacy_candidate_weight_histogram": legacy_candidate_weights,
        "legacy_effective_fraction_histogram": legacy_effective_fractions,
        "legacy_queries_without_candidate_aware": legacy_queries_without_candidate,
        "rejection_totals": rejection_totals,
        "weak_source_totals": weak_source_totals,
        "recall_source_totals": recall_source_totals,
    }


def load_stream_checkpoint(
    checkpoint_path: str | Path,
    checkpoint_contract: Mapping[str, object],
    output: str | Path,
    feature_output: str | Path,
) -> dict[str, object] | None:
    """Validate a retrain checkpoint and discard uncommitted tail parts."""
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    if checkpoint.get("artifact_type") != "final_retrain_checkpoint":
        raise ValueError("retrain checkpoint artifact type mismatch")
    checkpoint_version = checkpoint.get("artifact_version")
    if checkpoint_version not in {2, 3}:
        raise ValueError("retrain checkpoint version mismatch")
    if checkpoint.get("contract") != dict(checkpoint_contract):
        raise ValueError("retrain checkpoint contract mismatch")
    for dataset, part_key in (
        (Path(output).with_suffix(Path(output).suffix + ".tmp"), "pair_part_count"),
        (
            Path(feature_output).with_suffix(Path(feature_output).suffix + ".tmp"),
            "feature_part_count",
        ),
    ):
        if not dataset.is_dir():
            raise FileNotFoundError(f"checkpoint Parquet directory is missing: {dataset}")
        keep = int(checkpoint.get(part_key, -1))
        if keep < 0:
            raise ValueError("retrain checkpoint part count is invalid")
        for part in dataset.glob("part-*.parquet"):
            index = int(part.stem.split("-")[-1])
            if index >= keep:
                part.unlink()
        (dataset / "_SUCCESS").unlink(missing_ok=True)
    if checkpoint_version == 2:
        checkpoint["legacy_candidate_weight_histogram"] = checkpoint.pop(
            "candidate_weight_histogram", {}
        )
        checkpoint["legacy_effective_fraction_histogram"] = checkpoint.pop(
            "effective_fraction_histogram", {}
        )
        checkpoint["legacy_queries_without_candidate_aware"] = checkpoint.pop(
            "queries_without_candidate_aware", 0
        )
        checkpoint["loss_weight_shape_histogram"] = {}
    return checkpoint


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
    candidate_aware_fraction: float,
) -> dict[str, object]:
    totals = stats["totals"]
    assert isinstance(totals, Counter)
    rejection_totals = stats["rejection_totals"]
    weak_source_totals = stats["weak_source_totals"]
    recall_source_totals = stats["recall_source_totals"]
    if totals["negative_count"] != NEGATIVE_RATIO * totals["positive_count"]:
        raise ValueError("streamed training artifact violates the 1:3 ratio")
    effective_pair_count = int(stats.get("effective_pair_count", stats["pair_count"]))
    return {
        "artifact_type": "ranker_training_pairs",
        "artifact_version": TRAINING_PAIR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "stage": "final_retrain",
        "training_splits": sorted(training_splits("final_retrain")),
        "seed": PAIR_SEED,
        "random_sampling": "query_seeded_splitmix64_v1",
        "row_ordering": "positive_candidate_random_v1",
        "positive_selection": "recalled_weak_positives_only",
        "candidate_aware_sampling": CANDIDATE_SAMPLING_STRATEGY,
        "negative_ratio": NEGATIVE_RATIO,
        "candidate_aware_target_fraction": candidate_aware_fraction,
        "query_count": stats["query_count"],
        "pair_count": stats["pair_count"],
        "effective_pair_count": effective_pair_count,
        "dataset_layout": (
            "random_replacement_delta"
            if effective_pair_count != int(stats["pair_count"])
            else "materialized"
        ),
        "counts": dict(sorted(totals.items())),
        "actual_candidate_aware_fraction": (
            totals["candidate_aware_count"] / totals["negative_count"]
        ),
        "loss_weighting": _loss_weight_manifest(
            stats["loss_weight_totals"], candidate_aware_fraction, stats
        ),
        "rejection_counts": dict(sorted(rejection_totals.items())),
        "weak_positive_source_counts": dict(sorted(weak_source_totals.items())),
        "recall_source_totals": dict(sorted(recall_source_totals.items())),
        "candidate_pool_layout": "streamed_not_materialized",
        "negative_sampling_strategy": (
            "random_only_derived_from_full"
            if candidate_aware_fraction == 0.0
            else "candidate_aware_with_random_backfill"
        ),
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
    loss_weight_totals: Counter[str] = Counter()
    loss_weight_shape_histogram: Counter[str] = Counter()
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
            loss_weight_totals.update(audit["loss_weight_sums"])
            loss_weight_shape_histogram[_loss_weight_shape_key(audit)] += 1
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
        "loss_weighting": _loss_weight_manifest(
            loss_weight_totals,
            CANDIDATE_AWARE_FRACTION,
            {
                "loss_weight_shape_histogram": loss_weight_shape_histogram,
            },
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
        | StreamCheckpoint
        | StreamTableBatch
    ],
    output_path: str | Path,
    manifest_path: str | Path,
    feature_output_path: str | Path,
    feature_manifest_path: str | Path,
    *,
    parent_paths: Mapping[str, str | Path],
    scope: str,
    rows_per_file: int = 250_000,
    checkpoint_path: str | Path | None = None,
    checkpoint_contract: Mapping[str, object] | None = None,
    initial_checkpoint: Mapping[str, object] | None = None,
    candidate_aware_fraction: float = CANDIDATE_AWARE_FRACTION,
) -> tuple[dict[str, object], dict[str, object]]:
    """Write streamed retrain pairs and features in one partitioned pass."""
    if scope not in {"formal", "smoke"}:
        raise ValueError("training scope must be formal or smoke")
    output = Path(output_path)
    feature_output = Path(feature_output_path)
    stats = _write_streamed_rows(
        query_rows,
        output,
        feature_output,
        rows_per_file,
        candidate_aware_fraction=candidate_aware_fraction,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path is not None else None,
        checkpoint_contract=checkpoint_contract,
        initial_checkpoint=initial_checkpoint,
    )
    query_count = int(stats["query_count"])
    pair_count = int(stats["pair_count"])
    feature_count = int(stats["feature_count"])
    if query_count == 0 or pair_count == 0 or pair_count != feature_count:
        raise ValueError("streamed training pair and feature artifacts are inconsistent")
    parent_hashes = {
        name: sha256_path(path) for name, path in sorted(parent_paths.items())
    }
    manifest = _streamed_pair_manifest(
        output, stats, parent_hashes, scope, candidate_aware_fraction
    )
    write_json_atomic(manifest, manifest_path)
    feature_manifest = _streamed_feature_manifest(
        feature_output,
        stats,
        parent_hashes,
        output,
        manifest_path,
        scope,
        candidate_aware_fraction,
    )
    write_json_atomic(feature_manifest, feature_manifest_path)
    if checkpoint_path is not None:
        Path(checkpoint_path).unlink(missing_ok=True)
    return manifest, feature_manifest


def write_training_manifests_from_stats(
    output_path: str | Path,
    manifest_path: str | Path,
    feature_output_path: str | Path,
    feature_manifest_path: str | Path,
    *,
    stats: Mapping[str, object],
    parent_paths: Mapping[str, str | Path],
    scope: str,
    candidate_aware_fraction: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Publish manifests after an Arrow-native aligned dataset derivation."""
    output = Path(output_path)
    feature_output = Path(feature_output_path)
    if int(stats["pair_count"]) != int(stats["feature_count"]):
        raise ValueError("derived pair and feature counts differ")
    parent_hashes = {
        name: sha256_path(path) for name, path in sorted(parent_paths.items())
    }
    manifest = _streamed_pair_manifest(
        output,
        stats,
        parent_hashes,
        scope,
        candidate_aware_fraction,
    )
    write_json_atomic(manifest, manifest_path)
    feature_manifest = _streamed_feature_manifest(
        feature_output,
        stats,
        parent_hashes,
        output,
        manifest_path,
        scope,
        candidate_aware_fraction,
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
