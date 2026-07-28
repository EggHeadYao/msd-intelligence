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


def _macro(values: Mapping[str, float]) -> float:
    return fmean(values[group] for group in SPECIALIZATION_GROUPS)


def _passes_release_guards(
    values: Mapping[str, float],
    c1: Mapping[str, float],
    *,
    macro_margin: float,
) -> bool:
    return (
        values["audio_dominant"] >= AUDIO_RETENTION * c1["audio_dominant"]
        and values["relation_dominant"] > c1["relation_dominant"]
        and values["mixed"] >= c1["mixed"]
        and _macro(values) >= (1.0 + macro_margin) * _macro(c1)
    )


def select_guarded_ranker(
    tune_scores: Mapping[
        tuple[float, int, int, float, float], Mapping[str, Mapping[str, float]]
    ],
    confirm_scores: Mapping[
        tuple[float, int, int, float, float], Mapping[str, Mapping[str, float]]
    ],
    c1_tune_scores: Mapping[str, Mapping[str, float]],
    c1_confirm_scores: Mapping[str, Mapping[str, float]],
) -> tuple[tuple[float, int, int, float, float], dict[str, object]]:
    """Compatibility wrapper for callers that retain per-query scores."""
    return select_guarded_ranker_means(
        {key: _group_means(scores) for key, scores in tune_scores.items()},
        {key: _group_means(scores) for key, scores in confirm_scores.items()},
        _group_means(c1_tune_scores),
        _group_means(c1_confirm_scores),
    )


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
