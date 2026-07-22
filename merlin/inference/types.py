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


@dataclass(frozen=True, slots=True)
class RecallAudit:
    """Per-query candidate coverage diagnostics outside the ranker schema."""

    source_counts: Mapping[str, int]
    source_shortages: Mapping[str, int]
    unique_candidates: int
    raw_candidates: int = 0
    duplicate_candidates: int = 0
    deduplication_rate: float = 0.0
    exclusive_candidates: Mapping[str, int] = field(default_factory=dict)
    source_available: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_counts", MappingProxyType(dict(self.source_counts)))
        object.__setattr__(
            self,
            "source_shortages",
            MappingProxyType(dict(self.source_shortages)),
        )
        object.__setattr__(
            self,
            "exclusive_candidates",
            MappingProxyType(dict(self.exclusive_candidates)),
        )
        object.__setattr__(
            self,
            "source_available",
            MappingProxyType(dict(self.source_available)),
        )
        if self.raw_candidates < self.unique_candidates or self.duplicate_candidates < 0:
            raise ValueError("recall audit candidate counts are inconsistent")
        if not 0.0 <= self.deduplication_rate <= 1.0:
            raise ValueError("recall audit deduplication rate must be in [0, 1]")
