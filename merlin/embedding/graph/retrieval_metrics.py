"""Shared deterministic metrics for MERLIN retrieval experiments."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


DEFAULT_CUTOFFS = (10, 20, 50)


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in cutoffs)
    if not values or any(value <= 0 for value in values):
        raise ValueError("cutoffs must contain positive integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("cutoffs must be unique and increasing")
    return values


def _discount_sum(length: int) -> float:
    return sum(1.0 / math.log2(rank + 1.0) for rank in range(1, length + 1))


def score_ranking(
    ranked_track_ids: Sequence[str],
    positive_track_ids: Collection[str],
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, float]:
    """Score one filtered ranking with binary relevance."""
    cutoffs = _validate_cutoffs(cutoffs)
    ranked = tuple(str(track_id) for track_id in ranked_track_ids)
    positives = {str(track_id) for track_id in positive_track_ids}
    if not positives:
        raise ValueError("a scored query must have at least one positive")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked track IDs must be unique")

    result: dict[str, float] = {}
    first_positive_rank = next(
        (
            rank
            for rank, track_id in enumerate(ranked, start=1)
            if track_id in positives
        ),
        None,
    )
    result["mrr"] = 0.0 if first_positive_rank is None else 1.0 / first_positive_rank

    for cutoff in cutoffs:
        prefix = ranked[:cutoff]
        relevant = [track_id in positives for track_id in prefix]
        hits = sum(relevant)
        dcg = sum(
            1.0 / math.log2(rank + 1.0)
            for rank, is_relevant in enumerate(relevant, start=1)
            if is_relevant
        )
        ideal = _discount_sum(min(len(positives), cutoff))
        result[f"recall@{cutoff}"] = hits / len(positives)
        result[f"hit@{cutoff}"] = float(hits > 0)
        result[f"ndcg@{cutoff}"] = dcg / ideal
    return result
