"""Contracts between candidate generation, ranking, and reranking."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .types import Candidate


class CandidateRetriever(Protocol):
    """A Stage-1 nominator such as Audio FAISS, Graph FAISS, BFS, or Tag."""

    @property
    def name(self) -> str: ...

    def retrieve(self, query_track_id: str, limit: int) -> Sequence[Candidate]: ...


class PairFeatureComputer(Protocol):
    """Compute the versioned ranker features for one query-candidate pair."""

    @property
    def schema_version(self) -> str: ...

    def compute(self, query_track_id: str, candidate: Candidate) -> Mapping[str, float]: ...


class Ranker(Protocol):
    """Turn named pair features into one comparable relevance score."""

    @property
    def feature_schema_version(self) -> str: ...

    def score(self, features: Mapping[str, float]) -> float: ...
