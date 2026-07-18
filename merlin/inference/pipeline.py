"""End-to-end pure-Python MERLIN recommendation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .candidate_policy import validate_canonical_policy
from .interfaces import CandidateRetriever, PairFeatureComputer, Ranker
from .mmr import ScoredCandidate
from .retrieval import merge_candidates
from .types import Candidate, RecallAudit, Recommendation


@dataclass(slots=True)
class MerlinPipeline:
    retrievers: Sequence[CandidateRetriever]
    retriever_limits: Mapping[str, int]
    feature_computer: PairFeatureComputer
    ranker: Ranker
    final_limit: int = 20
    candidate_limit: int = 1_000
    canonical: bool = False

    def __post_init__(self) -> None:
        names = [retriever.name for retriever in self.retrievers]
        if len(set(names)) != len(names):
            raise ValueError("retriever names must be unique")
        missing_limits = [name for name in names if name not in self.retriever_limits]
        if missing_limits:
            raise ValueError(f"retriever limits missing: {missing_limits}")
        limits = [self.retriever_limits[name] for name in names]
        if any(limit <= 0 for limit in limits):
            raise ValueError("retriever limits must be positive")
        if self.candidate_limit <= 0 or sum(limits) > self.candidate_limit:
            raise ValueError("retriever limits exceed candidate union cap")
        if self.feature_computer.schema_version != self.ranker.feature_schema_version:
            raise ValueError("feature computer and ranker schema versions differ")
        if self.final_limit <= 0:
            raise ValueError("final_limit must be positive")
        if self.canonical:
            if set(names) != set(self.retriever_limits):
                raise ValueError("canonical pipeline retrievers and limits must match")
            validate_canonical_policy(
                self.retriever_limits,
                self.candidate_limit,
                self.final_limit,
            )

    def recommend(self, query_track_id: str, k: int | None = None) -> list[Recommendation]:
        """Recall candidates and rank the final tracks by LR raw margin."""
        if not query_track_id:
            raise ValueError("query_track_id must not be empty")
        final_limit = self.final_limit if k is None else k
        if final_limit <= 0 or final_limit > self.final_limit:
            raise ValueError("k must be between 1 and final_limit")

        candidates, _audit = self.recall(query_track_id)
        scored = []
        for candidate in candidates:
            features = dict(self.feature_computer.compute(query_track_id, candidate))
            scored.append(ScoredCandidate(candidate, self.ranker.score(features), features))
        ranked = sorted(
            scored,
            key=lambda item: (-item.relevance_score, item.candidate.track_id),
        )[:final_limit]
        return [
            Recommendation(
                track_id=item.candidate.track_id,
                relevance_score=item.relevance_score,
                rank=rank,
                sources=item.candidate.sources,
                features=item.features,
            )
            for rank, item in enumerate(ranked, start=1)
        ]

    def recall(self, query_track_id: str) -> tuple[list[Candidate], RecallAudit]:
        """Generate the canonical union and its per-source coverage audit."""
        if not query_track_id:
            raise ValueError("query_track_id must not be empty")
        groups = {
            retriever.name: list(retriever.retrieve(
                query_track_id, self.retriever_limits[retriever.name]
            ))
            for retriever in self.retrievers
        }
        candidates = merge_candidates(list(groups.values()), query_track_id)
        counts = {name: len(group) for name, group in groups.items()}
        shortages = {
            name: self.retriever_limits[name] - count
            for name, count in counts.items()
        }
        raw_count = sum(counts.values())
        unique_count = len(candidates)
        duplicates = raw_count - unique_count
        exclusive = {
            name: sum(candidate.sources == frozenset({name}) for candidate in candidates)
            for name in counts
        }
        audit = RecallAudit(
            source_counts=counts,
            source_shortages=shortages,
            unique_candidates=unique_count,
            raw_candidates=raw_count,
            duplicate_candidates=duplicates,
            deduplication_rate=duplicates / raw_count if raw_count else 0.0,
            exclusive_candidates=exclusive,
        )
        return candidates, audit
