"""End-to-end pure-Python MERLIN recommendation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .interfaces import CandidateRetriever, PairFeatureComputer, Ranker, RedundancyModel
from .mmr import ScoredCandidate, rerank_mmr
from .retrieval import merge_candidates
from .types import Recommendation


@dataclass(slots=True)
class MerlinPipeline:
    retrievers: Sequence[CandidateRetriever]
    retriever_limits: Mapping[str, int]
    feature_computer: PairFeatureComputer
    ranker: Ranker
    redundancy: RedundancyModel
    ranker_limit: int = 50
    final_limit: int = 20
    mmr_beta: float = 0.5

    def __post_init__(self) -> None:
        names = [retriever.name for retriever in self.retrievers]
        if len(set(names)) != len(names):
            raise ValueError("retriever names must be unique")
        missing_limits = [name for name in names if name not in self.retriever_limits]
        if missing_limits:
            raise ValueError(f"retriever limits missing: {missing_limits}")
        if self.feature_computer.schema_version != self.ranker.feature_schema_version:
            raise ValueError("feature computer and ranker schema versions differ")
        if self.ranker_limit <= 0 or self.final_limit <= 0 or self.final_limit > self.ranker_limit:
            raise ValueError("pipeline limits must satisfy 0 < final_limit <= ranker_limit")

    def recommend(self, query_track_id: str, k: int | None = None) -> list[Recommendation]:
        """Recall, score, and diversity-rerank recommendations for one song."""
        if not query_track_id:
            raise ValueError("query_track_id must not be empty")
        final_limit = self.final_limit if k is None else k
        if final_limit <= 0 or final_limit > self.ranker_limit:
            raise ValueError("k must be between 1 and ranker_limit")

        groups = [
            retriever.retrieve(query_track_id, self.retriever_limits[retriever.name])
            for retriever in self.retrievers
        ]
        candidates = merge_candidates(groups, query_track_id)
        scored = []
        for candidate in candidates:
            features = dict(self.feature_computer.compute(query_track_id, candidate))
            scored.append(ScoredCandidate(candidate, self.ranker.score(features), features))
        top = sorted(scored, key=lambda item: (-item.relevance_score, item.candidate.track_id))[
            : self.ranker_limit
        ]
        reranked = rerank_mmr(top, self.redundancy, final_limit, self.mmr_beta)
        return [
            Recommendation(
                track_id=item.candidate.track_id,
                relevance_score=item.relevance_score,
                rank=rank,
                sources=item.candidate.sources,
                features=item.features,
            )
            for rank, item in enumerate(reranked, start=1)
        ]
