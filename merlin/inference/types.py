"""Shared values passed between MERLIN inference components."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Candidate:
    """One deduplicated Stage-1 candidate and its recall evidence."""

    track_id: str
    sources: frozenset[str] = field(default_factory=frozenset)
    recall_scores: Mapping[str, float] = field(default_factory=dict)
    source_ranks: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("candidate track_id must not be empty")
        object.__setattr__(self, "recall_scores", MappingProxyType(dict(self.recall_scores)))
        object.__setattr__(self, "source_ranks", MappingProxyType(dict(self.source_ranks)))


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A final ranked recommendation returned by ``recommend``."""

    track_id: str
    relevance_score: float
    rank: int
    sources: frozenset[str]
    features: Mapping[str, float] = field(default_factory=dict)

