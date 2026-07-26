"""Deterministic query metrics and paired inference for Set-C evaluation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Iterable, Mapping, Sequence

from ..ranking.features import FEATURE_ORDER, FILL_FEATURES, materialize_raw_features
from ..ranking.model import LogisticRanker
from ..training.validation_groups import VALIDATION_QUERY_GROUPS
from .protocol import EVALUATION_CUTOFFS, EVALUATION_SEED


def retrieval_metrics(
    labels: Sequence[int],
    eligible_positive_count: int,
    *,
    cutoffs: Sequence[int] = EVALUATION_CUTOFFS,
) -> dict[str, float]:
    """Compute end-to-end binary retrieval metrics for one ranked query."""
    if eligible_positive_count <= 0:
        raise ValueError("eligible positive count must be positive")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("retrieval labels must be binary")
    result: dict[str, float] = {}
    first = next((rank for rank, label in enumerate(labels, 1) if label), None)
    result["mrr"] = 0.0 if first is None else 1.0 / first
    for cutoff in cutoffs:
        if cutoff <= 0:
            raise ValueError("retrieval cutoffs must be positive")
        top = labels[:cutoff]
        hits = sum(top)
        dcg = sum(
            label / math.log2(rank + 1.0)
            for rank, label in enumerate(top, 1)
        )
        idcg = sum(
            1.0 / math.log2(rank + 1.0)
            for rank in range(1, min(eligible_positive_count, cutoff) + 1)
        )
        result[f"recall@{cutoff}"] = hits / eligible_positive_count
        result[f"hit@{cutoff}"] = float(hits > 0)
        result[f"ndcg@{cutoff}"] = dcg / idcg
    return result


def random_ranking_expectation(
    candidate_count: int,
    recalled_positive_count: int,
    eligible_positive_count: int,
    *,
    cutoffs: Sequence[int] = EVALUATION_CUTOFFS,
) -> dict[str, float]:
    """Analytical random-order expectation for a query's actual pool size."""
    if not 0 <= recalled_positive_count <= candidate_count:
        raise ValueError("random expectation candidate-positive count is invalid")
    if candidate_count <= 0 or eligible_positive_count <= 0:
        raise ValueError("random expectation counts must be positive")
    result = {}
    survival = 1.0
    mrr = 0.0
    for rank in range(1, candidate_count + 1):
        remaining = candidate_count - rank + 1
        positive_probability = recalled_positive_count / remaining
        mrr += survival * positive_probability / rank
        survival *= (remaining - recalled_positive_count) / remaining
        if survival == 0.0:
            break
