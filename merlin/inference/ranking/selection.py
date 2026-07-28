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


def select_guarded_ranker_means(
    tune_scores: Mapping[
        tuple[float, int, int, float, float], Mapping[str, float]
    ],
    confirm_scores: Mapping[
        tuple[float, int, int, float, float], Mapping[str, float]
    ],
    c1_tune_scores: Mapping[str, float],
    c1_confirm_scores: Mapping[str, float],
) -> tuple[tuple[float, int, int, float, float], dict[str, object]]:
    """Select the strongest two-gate policy that passes both frozen folds."""
    if not tune_scores or set(tune_scores) != set(confirm_scores):
        raise ValueError("tune and confirmation configuration grids must align")
    if any(
        reg not in REG_PARAMS
        or middle_quota not in ADAPTIVE_AUDIO_QUOTAS
        or high_quota not in ADAPTIVE_AUDIO_QUOTAS
        or high_quota > middle_quota
        or low_threshold < 0.0
        or high_threshold < low_threshold
        for reg, middle_quota, high_quota, low_threshold, high_threshold in tune_scores
    ):
        raise ValueError("adaptive ranker configuration grid is invalid")
    tune = {
        key: _validated_group_means(scores) for key, scores in tune_scores.items()
    }
    confirm = {
        key: _validated_group_means(scores)
        for key, scores in confirm_scores.items()
    }
    c1_tune = _validated_group_means(c1_tune_scores)
    c1_confirm = _validated_group_means(c1_confirm_scores)
    tune_feasible = tuple(
        key for key, values in tune.items()
        if _passes_release_guards(values, c1_tune, macro_margin=0.0)
    )
    confirmed_feasible = tuple(
        key for key in tune_feasible
        if _passes_release_guards(
            confirm[key], c1_confirm, macro_margin=CONFIRM_MACRO_MARGIN
        )
    )
    fallback = max(
        (key for key in tune if key[1] == key[2] == 20),
        key=lambda key: (key[0], key[3], key[4]),
    )
    proposed = (
        max(
            confirmed_feasible,
            key=lambda key: (
                min(
                    _macro(tune[key]) / _macro(c1_tune),
                    _macro(confirm[key]) / _macro(c1_confirm),
                ),
                _macro(tune[key]) + _macro(confirm[key]),
                key[0],
                key[1],
                key[2],
                key[3],
                key[4],
            ),
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
