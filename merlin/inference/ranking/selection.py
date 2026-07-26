"""Frozen three-point LR selection with deterministic paired bootstrap ties."""

from __future__ import annotations

from statistics import fmean
from typing import Mapping, Sequence

import numpy as np


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
    if samples <= 0:
        raise ValueError("paired bootstrap sample count must be positive")
    differences = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    if not np.all(np.isfinite(differences)):
        raise ValueError("paired bootstrap inputs must be finite")
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    rows_per_chunk = max(1, min(samples, 1_000_000 // len(differences)))
    for start in range(0, samples, rows_per_chunk):
        end = min(start + rows_per_chunk, samples)
        indexes = generator.integers(
            0,
            len(differences),
            size=(end - start, len(differences)),
            dtype=np.int32,
        )
        bootstrap[start:end] = differences[indexes].mean(axis=1)
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
