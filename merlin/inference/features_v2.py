"""Ranker-v2 pair feature computation and metadata loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping


PairLookup = Callable[[str, str], float | None]


@dataclass(frozen=True, slots=True)
class TrackMetadataV2:
    """Metadata whose availability is meaningful to the v2 ranker."""

    release_id: str | None = None
    year: int | None = None


@dataclass(frozen=True, slots=True)
class PairSignalLookups:
    """Compute pair signals independently of candidate provenance."""

    audio: PairLookup
    graph: PairLookup
    bfs: PairLookup
    tags: PairLookup


@dataclass(frozen=True, slots=True)
class FeatureFillValues:
    """Set-A statistics used when a continuous pair signal is unavailable."""

    values: Mapping[str, float] = field(default_factory=dict)

    def get(self, feature_name: str) -> float:
        return float(self.values.get(feature_name, 0.0))
