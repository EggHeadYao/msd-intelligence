"""End-to-end pure-Python MERLIN recommendation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..recall.policy import validate_canonical_policy
from ..retrieval.faiss import FaissTrackIndex
from ..recall import recall_candidates
from ..types import (
    Candidate,
    CandidateRetriever,
    PairFeatureComputer,
    Ranker,
    RecallAudit,
    Recommendation,
)


@dataclass(frozen=True, slots=True)
class ColdAudioAudit:
    raw_results: int
    candidate_count: int
    candidate_shortage: int


@dataclass(slots=True)
class ColdAudioPipeline:
    audio_index: FaissTrackIndex
    track_to_song: Mapping[str, str] = field(default_factory=dict)
    candidate_limit: int = 1_000
    limit: int = 20
    overfetch_factor: int = 3

    def __post_init__(self) -> None:
        if self.candidate_limit != 1_000 or self.limit != 20:
            raise ValueError("cold Audio policy requires candidate_limit=1000 and limit=20")
        if self.overfetch_factor != 3:
            raise ValueError("cold Audio policy requires overfetch_factor=3")
        self.track_to_song = MappingProxyType(dict(self.track_to_song))

    def recommend(
        self,
        query_embedding: Any,
        k: int = 20,
        *,
        query_track_id: str | None = None,
        query_song_id: str | None = None,
    ) -> list[Recommendation]:
        recommendations, _audit = self.recommend_with_audit(
            query_embedding,
            k,
            query_track_id=query_track_id,
            query_song_id=query_song_id,
        )
        return recommendations

    def recommend_with_audit(
        self,
        query_embedding: Any,
        k: int = 20,
        *,
        query_track_id: str | None = None,
        query_song_id: str | None = None,
    ) -> tuple[list[Recommendation], ColdAudioAudit]:
        if not 1 <= k <= self.limit:
            raise ValueError("k must be between 1 and 20")
        requested = self.overfetch_factor * self.candidate_limit + 1
        raw = self.audio_index.search_vector(query_embedding, requested)
        filtered = []
        for track_id, score in raw:
            if track_id == query_track_id:
                continue
            if query_song_id is not None and self.track_to_song.get(track_id) == query_song_id:
                continue
            filtered.append((track_id, score))
            if len(filtered) == self.candidate_limit:
                break
        recommendations = [
            Recommendation(track_id, score, rank, frozenset({"audio"}), {"cos_audio": score})
            for rank, (track_id, score) in enumerate(filtered[:k], start=1)
        ]
        audit = ColdAudioAudit(len(raw), len(filtered), self.candidate_limit - len(filtered))
        return recommendations, audit


@dataclass(slots=True)
class MerlinPipeline:
    retrievers: Sequence[CandidateRetriever]
    retriever_limits: Mapping[str, int]
    feature_computer: PairFeatureComputer
    ranker: Ranker
    limit: int = 20
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
            scored.append((candidate, self.ranker.score(features), features))
        ranked = sorted(
            scored,
            key=lambda item: (-item[1], item[0].track_id),
        )[:final_limit]
        return [
            Recommendation(
                track_id=candidate.track_id,
                relevance_score=relevance_score,
                rank=rank,
                sources=candidate.sources,
                features=features,
            )
            for rank, (candidate, relevance_score, features) in enumerate(ranked, start=1)
        ]

    def recall(self, query_track_id: str) -> tuple[list[Candidate], RecallAudit]:
        """Generate the canonical union and its per-source coverage audit."""
        return recall_candidates(
            self.retrievers,
            self.retriever_limits,
            self.candidate_limit,
            query_track_id,
        )
