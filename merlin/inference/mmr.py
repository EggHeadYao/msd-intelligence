"""Maximum Marginal Relevance reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .interfaces import RedundancyModel
from .types import Candidate


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: Candidate
    relevance_score: float
    features: dict[str, float]


def rerank_mmr(
    candidates: Sequence[ScoredCandidate],
    redundancy: RedundancyModel,
    limit: int,
    beta: float = 0.5,
) -> list[ScoredCandidate]:
    if limit < 0 or beta < 0.0:
        raise ValueError("MMR limit and beta must be non-negative")
    remaining = list(candidates)
    selected: list[ScoredCandidate] = []
    while remaining and len(selected) < limit:
        def objective(item: ScoredCandidate) -> tuple[float, float, str]:
            penalty = max(
                (redundancy.similarity(item.candidate.track_id, prior.candidate.track_id)
                 for prior in selected),
                default=0.0,
            )
            return (item.relevance_score - beta * penalty, item.relevance_score, item.candidate.track_id)

        winner = max(remaining, key=objective)
        selected.append(winner)
        remaining.remove(winner)
    return selected
