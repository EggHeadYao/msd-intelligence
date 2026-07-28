"""Frozen Set-B selection for LR regularization and the Audio quota."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Mapping

REG_PARAMS = (0.001, 0.01, 0.1)
AUDIO_QUOTAS = tuple(range(21))
ADAPTIVE_AUDIO_QUOTAS = tuple(range(16, 21))
SPECIALIZATION_GROUPS = ("audio_dominant", "relation_dominant", "mixed")
AUDIO_RETENTION = 0.90
CONFIRM_MACRO_MARGIN = 0.01


def _group_means(
    grouped_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    if set(grouped_scores) != set(SPECIALIZATION_GROUPS) or any(
        not scores for scores in grouped_scores.values()
    ):
        raise ValueError("validation must contain exactly the three frozen strata")
    means = {
        group: fmean(float(value) for value in scores.values())
        for group, scores in grouped_scores.items()
    }
    if any(not math.isfinite(value) for value in means.values()):
        raise ValueError("validation scores must be finite")
    return means


def _validated_group_means(
    scores: Mapping[str, float],
) -> dict[str, float]:
    if set(scores) != set(SPECIALIZATION_GROUPS):
        raise ValueError("validation must contain exactly the three frozen strata")
    means = {group: float(scores[group]) for group in SPECIALIZATION_GROUPS}
    if any(not math.isfinite(value) for value in means.values()):
        raise ValueError("validation scores must be finite")
    return means


def select_reg_param(
    query_scores: Mapping[float, Sequence[float]],
) -> tuple[float, dict[str, object]]:
    if set(query_scores) != set(REG_PARAMS):
        raise ValueError("selection requires exactly the frozen three regParams")
    lengths = {len(values) for values in query_scores.values()}
    if len(lengths) != 1 or not lengths or 0 in lengths:
        raise ValueError("regParam query scores must be non-empty and aligned")
    means = {reg: fmean(values) for reg, values in query_scores.items()}
    reference = max(REG_PARAMS, key=lambda reg: (means[reg], reg))
    selected = reference
    comparisons = {}
    for larger in (reg for reg in REG_PARAMS if reg > reference):
        ci = paired_difference_ci(query_scores[reference], query_scores[larger])
        comparisons[f"{reference:g}_minus_{larger:g}"] = list(ci)
        if ci[0] <= 0.0 <= ci[1]:
            selected = max(selected, larger)
    return selected, {
        "mean_three_strata_ndcg20": {f"{reg:g}": means[reg] for reg in REG_PARAMS},
        "paired_difference_ci": comparisons,
        "selected_reg_param": selected,
        "tie_rule": "choose_larger_reg_param_when_paired_95pct_ci_contains_zero",
        "bootstrap_samples": SELECTION_BOOTSTRAPS,
        "seed": SELECTION_SEED,
    }


def _three_strata_mean(
    grouped_scores: Mapping[str, Mapping[str, float]],
) -> float:
    if not grouped_scores or any(not scores for scores in grouped_scores.values()):
        raise ValueError("three-strata scores must be non-empty")
    values = tuple(
        float(value)
        for scores in grouped_scores.values()
        for value in scores.values()
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("three-strata scores must be finite")
    return fmean(fmean(scores.values()) for scores in grouped_scores.values())


def paired_three_strata_difference_ci(
    left: Mapping[str, Mapping[str, float]],
    right: Mapping[str, Mapping[str, float]],
    *,
    samples: int = SELECTION_BOOTSTRAPS,
    seed: int = SELECTION_SEED,
) -> tuple[float, float]:
    """Bootstrap query IDs while giving each validation stratum equal weight."""
    if set(left) != set(right) or not left:
        raise ValueError("paired three-strata groups must be non-empty and aligned")
    if samples <= 0:
        raise ValueError("paired bootstrap sample count must be positive")
    for group in left:
        if set(left[group]) != set(right[group]) or not left[group]:
            raise ValueError("paired query scores must align within each stratum")
    query_ids = tuple(sorted({query_id for scores in left.values() for query_id in scores}))
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    differences = np.full((len(left), len(query_ids)), np.nan, dtype=np.float64)
    for group_index, group in enumerate(sorted(left)):
        for query_id, left_value in left[group].items():
            differences[group_index, query_index[query_id]] = (
                float(left_value) - float(right[group][query_id])
            )
    if not np.all(np.isfinite(differences[~np.isnan(differences)])):
        raise ValueError("paired bootstrap inputs must be finite")

    generator = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    rows_per_chunk = max(1, min(samples, 1_000_000 // len(query_ids)))
    written = 0
    while written < samples:
        batch_size = min(rows_per_chunk, samples - written)
        indexes = generator.integers(
            0,
            len(query_ids),
            size=(batch_size, len(query_ids)),
            dtype=np.int32,
        )
        group_means = np.empty((batch_size, len(left)), dtype=np.float64)
        valid = np.ones(batch_size, dtype=bool)
        for group_index in range(len(left)):
            sampled = differences[group_index, indexes]
            counts = np.count_nonzero(~np.isnan(sampled), axis=1)
            valid &= counts > 0
            group_means[:, group_index] = np.nansum(sampled, axis=1) / np.maximum(
                counts, 1
            )
        accepted = group_means[valid].mean(axis=1)
        take = min(len(accepted), samples - written)
        bootstrap[written : written + take] = accepted[:take]
        written += take
    return percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)


def select_grouped_reg_param(
    query_scores: Mapping[float, Mapping[str, Mapping[str, float]]],
) -> tuple[float, dict[str, object]]:
    """Select regParam from unequal eligible-query sets across frozen strata."""
    if set(query_scores) != set(REG_PARAMS):
        raise ValueError("selection requires exactly the frozen three regParams")
    group_sets = {tuple(sorted(scores)) for scores in query_scores.values()}
    if len(group_sets) != 1 or not next(iter(group_sets), ()):
        raise ValueError("regParam validation strata must be non-empty and aligned")
    reference_scores = query_scores[REG_PARAMS[0]]
    if any(
        set(scores[group]) != set(reference_scores[group])
        for scores in query_scores.values()
        for group in reference_scores
    ):
        raise ValueError("regParam query IDs must align within each stratum")
    means = {
        reg_param: _three_strata_mean(scores)
        for reg_param, scores in query_scores.items()
    }
    reference = max(REG_PARAMS, key=lambda reg: (means[reg], reg))
    selected = reference
    comparisons = {}
    for larger in (reg for reg in REG_PARAMS if reg > reference):
        ci = paired_three_strata_difference_ci(
            query_scores[reference], query_scores[larger]
        )
        comparisons[f"{reference:g}_minus_{larger:g}"] = list(ci)
        if ci[0] <= 0.0 <= ci[1]:
            selected = max(selected, larger)
    return selected, {
        "mean_three_strata_ndcg20": {
            f"{reg:g}": means[reg] for reg in REG_PARAMS
        },
        "paired_difference_ci": comparisons,
        "selected_reg_param": selected,
        "tie_rule": "choose_larger_reg_param_when_paired_95pct_ci_contains_zero",
        "bootstrap_samples": SELECTION_BOOTSTRAPS,
        "bootstrap_unit": "query_id_cluster_with_equal_stratum_means",
        "seed": SELECTION_SEED,
    }
