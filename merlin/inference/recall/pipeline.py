"""Ranker-independent canonical Stage-1 candidate recall."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from ..retrieval import merge_candidates
from ..types import Candidate, CandidateRetriever, RecallAudit
from .policy import validate_canonical_backfill, validate_canonical_policy


def allocate_backfill_groups(
    groups: Mapping[str, Sequence[Candidate]],
    primary_limits: Mapping[str, int],
    backfill_limits: Mapping[str, int],
    backfill_order: Sequence[str],
    candidate_limit: int,
) -> dict[str, list[Candidate]]:
    """Keep every primary nomination, then fill unused union capacity."""
    selected = {
        source: list(groups[source][: primary_limits[source]])
        for source in groups
    }
    used = {
        candidate.track_id
        for candidates in selected.values()
        for candidate in candidates
    }
    cursors = {source: primary_limits[source] for source in groups}
    while len(used) < candidate_limit:
        progressed = False
        for source in backfill_order:
            candidates = groups[source]
            boundary = min(len(candidates), backfill_limits[source])
            while cursors[source] < boundary:
                candidate = candidates[cursors[source]]
                cursors[source] += 1
                selected[source].append(candidate)
                if candidate.track_id not in used:
                    used.add(candidate.track_id)
                    progressed = True
                    break
            if len(used) >= candidate_limit:
                break
        if not progressed:
            break
    return selected


def validate_recall_configuration(
    retrievers: Sequence[CandidateRetriever],
    retriever_limits: Mapping[str, int],
    candidate_limit: int,
    *,
    canonical: bool,
) -> None:
    names = [retriever.name for retriever in retrievers]
    if not names or len(set(names)) != len(names):
        raise ValueError("recall retriever names must be non-empty and unique")
    if set(names) != set(retriever_limits):
        raise ValueError("recall retrievers and limits must match")
    limits = [int(retriever_limits[name]) for name in names]
    if any(limit <= 0 for limit in limits):
        raise ValueError("recall limits must be positive")
    if candidate_limit <= 0 or sum(limits) > candidate_limit:
        raise ValueError("retriever limits exceed candidate union cap")
    if canonical:
        validate_canonical_policy(retriever_limits, candidate_limit, 20)


def recall_candidates(
    retrievers: Sequence[CandidateRetriever],
    retriever_limits: Mapping[str, int],
    candidate_limit: int,
    query_track_id: str,
    backfill_limits: Mapping[str, int] | None = None,
    backfill_order: Sequence[str] = (),
) -> tuple[list[Candidate], RecallAudit]:
    """Generate one deterministic candidate union and coverage audit."""
    if not query_track_id:
        raise ValueError("query_track_id must not be empty")
    groups: dict[str, list[Candidate]] = {}
    availability: dict[str, bool] = {}
    for retriever in retrievers:
        available = getattr(retriever, "is_available", lambda _query: True)
        availability[retriever.name] = bool(available(query_track_id))
        groups[retriever.name] = (
            list(retriever.retrieve(
                query_track_id,
                (backfill_limits or retriever_limits)[retriever.name],
            ))
            if availability[retriever.name]
            else []
        )

    if backfill_limits is not None:
        groups = allocate_backfill_groups(
            groups,
            retriever_limits,
            backfill_limits,
            backfill_order,
            candidate_limit,
        )
    return audit_recall_groups(
        groups,
        retriever_limits,
        candidate_limit,
        query_track_id,
        availability,
    )


def audit_recall_groups(
    groups: Mapping[str, Sequence[Candidate]],
    retriever_limits: Mapping[str, int],
    candidate_limit: int,
    query_track_id: str,
    availability: Mapping[str, bool],
) -> tuple[list[Candidate], RecallAudit]:
    """Merge independently generated source groups into one canonical audit."""
    if set(groups) != set(retriever_limits) or set(availability) != set(groups):
        raise ValueError("recall groups, limits, and availability must match")
    candidates = merge_candidates(list(groups.values()), query_track_id)
    if len(candidates) > candidate_limit:
        raise ValueError("candidate union exceeds configured cap")
    counts = {name: len(group) for name, group in groups.items()}
    shortages = {
        name: max(0, int(retriever_limits[name]) - count)
        for name, count in counts.items()
    }
    raw_count = sum(counts.values())
    unique_count = len(candidates)
    duplicates = raw_count - unique_count
    exclusive = {
        name: sum(candidate.sources == frozenset({name}) for candidate in candidates)
        for name in counts
    }
    return candidates, RecallAudit(
        source_counts=counts,
        source_shortages=shortages,
        unique_candidates=unique_count,
        raw_candidates=raw_count,
        duplicate_candidates=duplicates,
        deduplication_rate=duplicates / raw_count if raw_count else 0.0,
        exclusive_candidates=exclusive,
        source_available=availability,
    )


@dataclass(slots=True)
class RecallPipeline:
    """The canonical four-source candidate generator without a Ranker."""

    retrievers: Sequence[CandidateRetriever]
    retriever_limits: Mapping[str, int]
    candidate_limit: int = 1_000
    canonical: bool = False
    backfill_limits: Mapping[str, int] | None = None
    backfill_order: Sequence[str] = ()

    def __post_init__(self) -> None:
        validate_recall_configuration(
            self.retrievers,
            self.retriever_limits,
            self.candidate_limit,
            canonical=self.canonical,
        )
        if self.backfill_limits is not None:
            if set(self.backfill_limits) != set(self.retriever_limits):
                raise ValueError("backfill limits must cover every retriever")
            if any(
                int(self.backfill_limits[name]) < int(limit)
                for name, limit in self.retriever_limits.items()
            ):
                raise ValueError("backfill limits must retain every primary quota")
            if set(self.backfill_order) - set(self.retriever_limits):
                raise ValueError("backfill order contains an unknown source")
        if self.canonical:
            if self.backfill_limits is None:
                raise ValueError("canonical recall requires a backfill policy")
            validate_canonical_backfill(
                self.backfill_limits,
                tuple(self.backfill_order),
            )

    def recall(self, query_track_id: str) -> tuple[list[Candidate], RecallAudit]:
        return recall_candidates(
            self.retrievers,
            self.retriever_limits,
            self.candidate_limit,
            query_track_id,
            self.backfill_limits,
            self.backfill_order,
        )

    def recall_many(
        self,
        query_track_ids: Sequence[str],
        *,
        source_overrides: Mapping[str, Mapping[str, Sequence[Candidate]]] | None = None,
    ) -> Mapping[str, tuple[list[Candidate], RecallAudit]]:
        """Generate a bounded batch while allowing vector retrievers to batch-search."""
        queries = tuple(query_track_ids)
        if any(not query_id for query_id in queries):
            raise ValueError("batch query track IDs must not be empty")
        if len(set(queries)) != len(queries):
            raise ValueError("batch query track IDs must be unique")
        groups: dict[str, dict[str, Sequence[Candidate]]] = {
            query_id: {} for query_id in queries
        }
        availability = {query_id: {} for query_id in queries}
        overrides = source_overrides or {}
        unknown_sources = set(overrides) - set(self.retriever_limits)
        if unknown_sources:
            raise ValueError(f"recall overrides contain unknown sources: {sorted(unknown_sources)}")
        for retriever in self.retrievers:
            available = getattr(retriever, "is_available", lambda _query: True)
            states = {query_id: bool(available(query_id)) for query_id in queries}
            retrieve_many = getattr(retriever, "retrieve_many", None)
            retrieved = overrides.get(retriever.name)
            if retrieved is None:
                retrieved = (
                    retrieve_many(
                        queries,
                        (self.backfill_limits or self.retriever_limits)[retriever.name],
                    )
                    if retrieve_many is not None
                    else {
                        query_id: retriever.retrieve(
                            query_id,
                            (self.backfill_limits or self.retriever_limits)[retriever.name],
                        )
                        for query_id in queries
                        if states[query_id]
                    }
                )
            for query_id in queries:
                availability[query_id][retriever.name] = states[query_id]
                groups[query_id][retriever.name] = retrieved.get(query_id, ())
        return {
            query_id: audit_recall_groups(
                (
                    allocate_backfill_groups(
                        groups[query_id],
                        self.retriever_limits,
                        self.backfill_limits,
                        self.backfill_order,
                        self.candidate_limit,
                    )
                    if self.backfill_limits is not None
                    else groups[query_id]
                ),
                self.retriever_limits,
                self.candidate_limit,
                query_id,
                availability[query_id],
            )
            for query_id in queries
        }


def iter_recalled_candidates(
    pipeline: RecallPipeline,
    query_track_ids: Iterable[str],
    *,
    batch_size: int = 256,
) -> Iterator[tuple[str, list[Candidate], RecallAudit]]:
    """Yield canonical candidates without materializing or persisting the pool."""
    if batch_size <= 0:
        raise ValueError("recall batch size must be positive")
    iterator = iter(query_track_ids)
    previous_query: str | None = None
    while True:
        batch: list[str] = []
        for _ in range(batch_size):
            query_id = next(iterator, None)
            if query_id is None:
                break
            if not query_id:
                raise ValueError("query track IDs must not be empty")
            if previous_query is not None and query_id <= previous_query:
                raise ValueError("streaming recall queries must be strictly sorted")
            previous_query = query_id
            batch.append(query_id)
        if not batch:
            return
        recalled = pipeline.recall_many(batch)
        for query_id in batch:
            candidates, audit = recalled[query_id]
            yield query_id, candidates, audit


def candidate_digest(candidates: Sequence[Candidate]) -> str:
    payload = [
        {
            "track_id": candidate.track_id,
            "sources": sorted(candidate.sources),
            "recall_scores": dict(sorted(candidate.recall_scores.items())),
            "source_ranks": dict(sorted(candidate.source_ranks.items())),
        }
        for candidate in candidates
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def recall_query_report(
    query_id: str,
    candidates: Sequence[Candidate],
    audit: RecallAudit,
) -> dict[str, object]:
    return {
        "query_track_id": query_id,
        "candidate_digest_sha256": candidate_digest(candidates),
        "raw_candidates": audit.raw_candidates,
        "unique_candidates": audit.unique_candidates,
        "duplicate_candidates": audit.duplicate_candidates,
        "deduplication_rate": audit.deduplication_rate,
        "source_available": dict(audit.source_available),
        "source_counts": dict(audit.source_counts),
        "source_shortages": dict(audit.source_shortages),
        "exclusive_candidates": dict(audit.exclusive_candidates),
    }


def validate_recall_pipeline(
    pipeline: RecallPipeline,
    query_track_ids: Iterable[str],
) -> dict[str, object]:
    """Repeat fixed queries and report deterministic structural recall coverage."""
    reports: list[dict[str, object]] = []
    for query_id in query_track_ids:
        first, audit = pipeline.recall(query_id)
        second, repeated_audit = pipeline.recall(query_id)
        first_digest = candidate_digest(first)
        if first_digest != candidate_digest(second) or audit != repeated_audit:
            raise ValueError(f"recall is not deterministic for {query_id}")
        reports.append(recall_query_report(query_id, first, audit))
    if not reports:
        raise ValueError("recall validation requires at least one query")
    return {
        "validation_status": "PASS",
        "validation_type": "structural_recall_audit",
        "candidate_recall_metrics_available": False,
        "query_count": len(reports),
        "queries": reports,
    }


def write_recall_report(report: Mapping[str, object], path: str | Path) -> None:
    """Atomically publish a structural recall validation report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(report), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)
