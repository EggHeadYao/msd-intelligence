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
    result["mrr"] = mrr
    for cutoff in cutoffs:
        effective = min(cutoff, candidate_count)
        expected_hits = effective * recalled_positive_count / candidate_count
        no_hit = 1.0
        for offset in range(effective):
            no_hit *= (
                candidate_count - recalled_positive_count - offset
            ) / (candidate_count - offset)
            if no_hit == 0.0:
                break
        expected_dcg = recalled_positive_count / candidate_count * sum(
            1.0 / math.log2(rank + 1.0) for rank in range(1, effective + 1)
        )
        idcg = sum(
            1.0 / math.log2(rank + 1.0)
            for rank in range(1, min(eligible_positive_count, cutoff) + 1)
        )
        result[f"recall@{cutoff}"] = expected_hits / eligible_positive_count
        result[f"hit@{cutoff}"] = 1.0 - no_hit
        result[f"ndcg@{cutoff}"] = expected_dcg / idcg
    return result


def stable_random_scores(query_id: str, size: int, seed: int = EVALUATION_SEED):
    """Return a reproducible random scorer without Python hash randomization."""
    import numpy as np

    digest = hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return generator.random(size)


def score_query(
    query_id: str,
    rows: Sequence[Mapping[str, object]],
    *,
    full_ranker: LogisticRanker,
    no_hard_ranker: LogisticRanker,
    fill_values: Mapping[str, float],
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    """Score one canonical candidate list with all frozen baselines."""
    import numpy as np

    if not rows:
        raise ValueError("Set-C query candidate list must not be empty")
    candidate_ids = np.asarray(
        [str(row["candidate_track_id"]) for row in rows], dtype=object
    )
    if len(set(candidate_ids.tolist())) != len(candidate_ids):
        raise ValueError(f"duplicate canonical candidate for query {query_id}")
    materialized = [materialize_raw_features(row, fill_values) for row in rows]
    matrix = np.asarray(
        [[features[name] for name in FEATURE_ORDER] for features in materialized],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"non-finite Set-C feature for query {query_id}")

    audio_index = FEATURE_ORDER.index("cos_audio")
    full_means = np.asarray(full_ranker.means, dtype=np.float64)
    full_stds = np.asarray(full_ranker.stds, dtype=np.float64)
    full_scaled = (matrix - full_means) / full_stds
    if (
        no_hard_ranker.means == full_ranker.means
        and no_hard_ranker.stds == full_ranker.stds
    ):
        no_hard_scaled = full_scaled
    else:
        no_hard_scaled = (
