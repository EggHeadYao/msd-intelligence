"""Deterministic query metrics and paired development evaluation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Iterable, Mapping, Sequence

from ..ranking.features import FEATURE_ORDER, FILL_FEATURES, materialize_raw_features
from ..ranking.model import (
    LogisticRanker,
    query_relation_evidence,
    quota_interleave_ordered_indices,
)
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
        raise ValueError("development query candidate list must not be empty")
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
        raise ValueError(f"non-finite development feature for query {query_id}")

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
            matrix - np.asarray(no_hard_ranker.means, dtype=np.float64)
        ) / np.asarray(no_hard_ranker.stds, dtype=np.float64)
    def ranker_scores(ranker: LogisticRanker, scaled):
        return (
            scaled @ np.asarray(ranker.coefficients, dtype=np.float64)
            + ranker.intercept
        )

    raw = {name: matrix[:, index] for index, name in enumerate(FEATURE_ORDER)}
    graph = np.where(raw["has_graph"] > 0.0, raw["cos_graph"], -np.inf)
    bfs = np.where(raw["has_bfs"] > 0.0, raw["bfs_score"], -np.inf)
    scores = {
        "full": ranker_scores(full_ranker, full_scaled),
        "random": stable_random_scores(query_id, len(rows)),
        "c1_only": raw["cos_audio"],
        "c2_only": graph,
        "handcrafted": np.where(
            raw["has_graph"] > 0.0,
            0.5 * raw["cos_audio"] + 0.5 * raw["cos_graph"],
            raw["cos_audio"],
        ),
        "bfs": bfs,
        "no_hard_neg": ranker_scores(
            no_hard_ranker,
            no_hard_scaled,
        ),
    }
    score_orders = {
        name: np.lexsort((candidate_ids, -values))
        for name, values in scores.items()
    }
    rankings = {
        name: candidate_ids[order].tolist()
        for name, order in score_orders.items()
    }
    relation_evidence = query_relation_evidence(materialized)
    audio_order = score_orders["c1_only"]
    for name, ranker in (
        ("full", full_ranker),
        ("no_hard_neg", no_hard_ranker),
    ):
        effective_quota = ranker.effective_audio_quota(relation_evidence)
        learned_order = score_orders[name]
        indexes = quota_interleave_ordered_indices(
            learned_order,
            audio_order,
            effective_quota,
        )
        selected = set(indexes)
        complete = (*indexes, *(int(index) for index in learned_order if index not in selected))
        rankings[name] = candidate_ids[list(complete)].tolist()
    cold_positions = np.asarray([
        index
        for index, row in enumerate(rows)
        if any(source != "audio" for source in row.get("recall_sources", ()))
    ], dtype=np.int64)
    cold_matrix = matrix[cold_positions].copy()
    interaction_index = FEATURE_ORDER.index("audio_tag_interaction")
    tag_index = FEATURE_ORDER.index("tag_tfidf_cosine")
    cold_matrix[:, audio_index] = float(fill_values["cos_audio"])
    cold_matrix[:, interaction_index] = (
        float(fill_values["cos_audio"]) * cold_matrix[:, tag_index]
    )
    cold_scaled = (cold_matrix - full_means) / full_stds
    cold_scores = ranker_scores(full_ranker, cold_scaled)
    cold_ids = candidate_ids[cold_positions]
    rankings["precomputed_acoustic_cold"] = cold_ids[
        np.lexsort((cold_ids, -cold_scores))
    ].tolist()
    group_labels: dict[str, dict[str, int]] = defaultdict(dict)
    group_denominators: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        candidate_id = str(row["candidate_track_id"])
        for membership in row["validation_groups"]:
            group = str(membership["query_group"])
            group_labels[group][candidate_id] = int(membership["label"])
            group_denominators[group].add(
                int(membership["eligible_positive_count"])
            )
    query_metrics: list[dict[str, object]] = []
    for scorer, ranking in rankings.items():
        for group in VALIDATION_QUERY_GROUPS:
            labels_by_id = group_labels.get(group, {})
            if not labels_by_id:
                continue
            missing = set(ranking).difference(labels_by_id)
            if missing:
                raise ValueError(
                    f"validation labels are incomplete for query {query_id}, "
                    f"group {group}"
                )
            labels = [labels_by_id[candidate_id] for candidate_id in ranking]
            eligible_counts = group_denominators[group]
            if len(eligible_counts) != 1:
                raise ValueError("eligible-positive denominator changed within query")
            eligible = next(iter(eligible_counts))
            query_metrics.append({
                "query_track_id": query_id,
                "query_group": group,
                "scorer": scorer,
                "candidate_count": len(labels),
                "eligible_positive_count": eligible,
                **retrieval_metrics(labels, eligible),
            })
    return query_metrics, rankings


def macro_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Average query metrics per group and with equal three-strata weight."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_group"])].append(row)
    if set(grouped) != set(VALIDATION_QUERY_GROUPS):
        raise ValueError("macro metrics require all frozen validation groups")
    metric_names = (
        *(f"{name}@{cutoff}" for cutoff in EVALUATION_CUTOFFS for name in ("recall", "hit", "ndcg")),
        "mrr",
    )
    by_group = {}
    for group in VALIDATION_QUERY_GROUPS:
        values = grouped[group]
        by_group[group] = {
            "query_count": len(values),
            **{
                name: sum(float(row[name]) for row in values) / len(values)
                for name in metric_names
            },
        }
    return {
        "by_group": by_group,
        "three_strata_macro": {
            name: sum(float(by_group[group][name]) for group in VALIDATION_QUERY_GROUPS)
            / len(VALIDATION_QUERY_GROUPS)
            for name in metric_names
        },
    }


def paired_bootstrap_ci(
    rows: Sequence[Mapping[str, object]],
    *,
    baseline: str,
    metric: str,
    samples: int,
    seed: int = EVALUATION_SEED,
    clusters: Mapping[str, str] | None = None,
) -> dict[str, float | int]:
    """Bootstrap Full-minus-baseline with equal validation-stratum weight."""
    import numpy as np

    values = {
        (str(row["query_track_id"]), str(row["query_group"]), str(row["scorer"])):
        float(row[metric])
        for row in rows
        if row["scorer"] in {"full", baseline}
    }
    differences: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for query_id, group, scorer in values:
        if scorer != "full":
            continue
        other = values.get((query_id, group, baseline))
        if other is not None:
            differences[group].append((query_id, values[(query_id, group, scorer)] - other))
    if any(not differences[group] for group in VALIDATION_QUERY_GROUPS):
        raise ValueError(f"paired bootstrap is missing rows for {baseline}")
    if samples <= 0:
        raise ValueError("paired bootstrap samples must be positive")
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    if clusters is None:
        arrays = {
            group: np.asarray([value for _query, value in differences[group]])
            for group in VALIDATION_QUERY_GROUPS
        }
        for start in range(0, samples, 64):
            stop = min(start + 64, samples)
            block = np.zeros(stop - start, dtype=np.float64)
            for group_values in arrays.values():
                indexes = generator.integers(
                    0,
                    len(group_values),
                    size=(stop - start, len(group_values)),
                )
                block += group_values[indexes].mean(axis=1)
            estimates[start:stop] = block / len(VALIDATION_QUERY_GROUPS)
    else:
        cluster_rows: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for group, items in differences.items():
            for query_id, value in items:
                cluster_rows[clusters.get(query_id, f"missing:{query_id}")][group].append(value)
        cluster_ids = tuple(sorted(cluster_rows))
        sums = np.zeros((len(cluster_ids), len(VALIDATION_QUERY_GROUPS)))
        counts = np.zeros_like(sums)
        for cluster_index, cluster_id in enumerate(cluster_ids):
            for group_index, group in enumerate(VALIDATION_QUERY_GROUPS):
                group_values = cluster_rows[cluster_id].get(group, ())
                sums[cluster_index, group_index] = sum(group_values)
                counts[cluster_index, group_index] = len(group_values)
        for start in range(0, samples, 64):
            stop = min(start + 64, samples)
            indexes = generator.integers(
                0,
                len(cluster_ids),
                size=(stop - start, len(cluster_ids)),
            )
            sampled_sums = sums[indexes].sum(axis=1)
            sampled_counts = counts[indexes].sum(axis=1)
            if np.any(sampled_counts == 0):
                raise ValueError("artist bootstrap sample omitted a validation group")
            estimates[start:stop] = (sampled_sums / sampled_counts).mean(axis=1)
    point = sum(
        sum(value for _query, value in differences[group]) / len(differences[group])
        for group in VALIDATION_QUERY_GROUPS
    ) / len(VALIDATION_QUERY_GROUPS)
    low, high = np.quantile(estimates, (0.025, 0.975), method="linear")
    return {
        "samples": samples,
        "seed": seed,
        "point_difference": float(point),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }
