from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


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


