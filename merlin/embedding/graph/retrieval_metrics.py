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


def random_expectation(
    catalog_candidates: int,
    positive_count: int,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, float]:
    """Return exact expectations for a uniform ranking without replacement."""
    cutoffs = _validate_cutoffs(cutoffs)
    population = int(catalog_candidates)
    positives = int(positive_count)
    if population <= 0:
        raise ValueError("catalog candidate count must be positive")
    if positives <= 0 or positives > population:
        raise ValueError("positive count must be within the candidate catalog")

    result: dict[str, float] = {}
    max_rank = min(cutoffs[-1], population)
    no_positive_before = 1.0
    expected_rr = 0.0
    for rank in range(1, max_rank + 1):
        remaining = population - rank + 1
        first_at_rank = no_positive_before * positives / remaining
        expected_rr += first_at_rank / rank
        non_positive_remaining = population - positives - rank + 1
        if non_positive_remaining <= 0:
            no_positive_before = 0.0
        else:
            no_positive_before *= non_positive_remaining / remaining
    result["mrr"] = expected_rr

    for cutoff in cutoffs:
        k = min(cutoff, population)
        expected_recall = k / population
        no_hit = 1.0
        for draw in range(k):
            non_positive_remaining = population - positives - draw
            total_remaining = population - draw
            if non_positive_remaining <= 0:
                no_hit = 0.0
                break
            no_hit *= non_positive_remaining / total_remaining
        expected_dcg = positives / population * _discount_sum(k)
        ideal = _discount_sum(min(positives, cutoff))
        result[f"recall@{cutoff}"] = expected_recall
        result[f"hit@{cutoff}"] = 1.0 - no_hit
        result[f"ndcg@{cutoff}"] = expected_dcg / ideal
    return result
