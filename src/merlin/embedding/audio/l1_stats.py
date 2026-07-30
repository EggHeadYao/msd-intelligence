from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def classify_validation(
    pair_counts: Mapping[str, int],
    pair_types: Sequence[str],
    target_count: int,
    allow_partial_pairs: bool,
    supported: bool,
) -> tuple[bool, str]:
    formal = not allow_partial_pairs and all(
        pair_counts.get(pair_type) == target_count for pair_type in pair_types
    )
    if not formal:
        return False, "SMOKE_PASS"
    return True, "PASS" if supported else "FAIL"


def finite_array(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sample")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return result


def distribution(values: Sequence[float]) -> dict[str, Any]:
    sample = finite_array(values, "distribution")
    quantiles = np.quantile(sample, QUANTILES)
    return {
        "count": int(sample.size),
        "mean": float(np.mean(sample)),
        "median": float(np.median(sample)),
        "stddev": float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0,
        "min": float(np.min(sample)),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(np.max(sample)),
    }


def hedges_g(related: Sequence[float], random_sample: Sequence[float]) -> float | None:
    related_array = finite_array(related, "related sample")
    random_array = finite_array(random_sample, "random sample")
    degrees = related_array.size + random_array.size - 2
    if degrees <= 0:
        return None
    related_ss = float(np.sum((related_array - np.mean(related_array)) ** 2))
    random_ss = float(np.sum((random_array - np.mean(random_array)) ** 2))
    pooled = math.sqrt((related_ss + random_ss) / degrees)
    if pooled == 0.0:
        return None
    correction = 1.0 - 3.0 / (4.0 * degrees - 1.0) if degrees > 1 else 1.0
    return float(correction * (np.mean(related_array) - np.mean(random_array)) / pooled)


def bootstrap_hedges_g_ci(
    related: Sequence[float],
    random_sample: Sequence[float],
    samples: int,
    seed: int,
) -> list[float] | None:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    # Spark collection order is not stable. Canonical sorting makes a fixed seed
    # reproduce the same bootstrap draws for the same multiset of scores.
    related_array = np.sort(finite_array(related, "related sample"))
    random_array = np.sort(finite_array(random_sample, "random sample"))
    if related_array.size + random_array.size <= 2:
        return None
    rng = np.random.default_rng(seed)
    effects: list[np.ndarray] = []
    remaining = samples
    batch_size = 50
    degrees = related_array.size + random_array.size - 2
    correction = 1.0 - 3.0 / (4.0 * degrees - 1.0) if degrees > 1 else 1.0
    while remaining:
        batch = min(batch_size, remaining)
        related_draws = related_array[
            rng.integers(0, related_array.size, size=(batch, related_array.size))
        ]
        random_draws = random_array[
            rng.integers(0, random_array.size, size=(batch, random_array.size))
        ]
        related_means = np.mean(related_draws, axis=1)
        random_means = np.mean(random_draws, axis=1)
        related_ss = np.sum((related_draws - related_means[:, None]) ** 2, axis=1)
        random_ss = np.sum((random_draws - random_means[:, None]) ** 2, axis=1)
        pooled = np.sqrt((related_ss + random_ss) / degrees)
        valid = pooled > 0.0
        if np.any(valid):
            effects.append(
                correction * (related_means[valid] - random_means[valid]) / pooled[valid]
            )
        remaining -= batch
    if not effects:
        return None
    values = np.concatenate(effects)
    low, high = np.quantile(values, (0.025, 0.975))
    return [float(low), float(high)]


def preservation_summary(
    before: Sequence[float],
    after: Sequence[float],
) -> dict[str, float | int | None]:
    before_array = finite_array(before, "pre-PCA sample")
    after_array = finite_array(after, "PCA sample")
    if before_array.size != after_array.size:
        raise ValueError("pre-PCA and PCA samples must contain the same pairs")
    correlation: float | None = None
    if before_array.size > 1 and np.std(before_array) > 0.0 and np.std(after_array) > 0.0:
        correlation = float(np.corrcoef(before_array, after_array)[0, 1])
    return {
        "count": int(before_array.size),
        "pearson_correlation": correlation,
        "mean_absolute_similarity_delta": float(np.mean(np.abs(after_array - before_array))),
        "mean_signed_similarity_delta": float(np.mean(after_array - before_array)),
    }
