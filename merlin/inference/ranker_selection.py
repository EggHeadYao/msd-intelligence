"""Frozen three-point LR selection with deterministic paired bootstrap ties."""

from __future__ import annotations

import random
from statistics import fmean
from typing import Mapping, Sequence


REG_PARAMS = (0.001, 0.01, 0.1)
SELECTION_BOOTSTRAPS = 2_000
SELECTION_SEED = 42


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile values must not be empty")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_difference_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = SELECTION_BOOTSTRAPS,
    seed: int = SELECTION_SEED,
) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap inputs must be non-empty and aligned")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    generator = random.Random(seed)
    bootstrap = [
        fmean(differences[generator.randrange(len(differences))] for _ in differences)
        for _sample in range(samples)
    ]
    return percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)


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
