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
        if confirmed_feasible
        else fallback
    )
    confirmed = bool(confirmed_feasible)
    selected = proposed
    return selected, {
        "selection_metric": "equal_three_strata_query_macro_ndcg@20",
        "selected_reg_param": selected[0],
        "selected_audio_quota": selected[1],
        "selected_high_evidence_audio_quota": selected[2],
        "selected_relation_gate_threshold": selected[3],
        "selected_high_relation_gate_threshold": selected[4],
        "publishable_fusion": confirmed,
        "fallback_policy": "c1_only",
        "proposed_configuration": {
            "reg_param": proposed[0],
            "audio_quota": proposed[1],
            "high_evidence_audio_quota": proposed[2],
            "relation_gate_threshold": proposed[3],
            "high_relation_gate_threshold": proposed[4],
        },
        "tune_metrics": tune[proposed],
        "confirmation_metrics": confirm[proposed],
        "c1_tune_metrics": c1_tune,
        "c1_confirmation_metrics": c1_confirm,
        "release_guards": {
            "audio_retention": AUDIO_RETENTION,
            "relation_strictly_above_c1": True,
            "mixed_not_below_c1": True,
            "confirmation_macro_relative_margin": CONFIRM_MACRO_MARGIN,
        },
        "tune_feasible_configuration_count": len(tune_feasible),
        "confirmed_feasible_configuration_count": len(confirmed_feasible),
        "tie_rule": "higher_worst_fold_gain_then_higher_combined_macro_then_safer_quota",
        "configuration_count": len(tune),
    }
