# ruff: noqa: T201
"""Extract segment arrays, similar artists, and artist terms from per-song .h5 files."""

from __future__ import annotations

import numpy as np


def aggregate_segments(
    pitches: np.ndarray,
    timbre: np.ndarray,
    loudness_max: np.ndarray,
) -> np.ndarray:
    """Aggregate variable-length segment arrays into a fixed 99-dim vector.

    Args:
        pitches: Nx12 float64 array from /analysis/segments_pitches.
        timbre: Nx12 float64 array from /analysis/segments_timbre.
        loudness_max: N float64 array from /analysis/segments_loudness_max.

    Returns:
        1-D numpy array of length 99 (48 + 48 + 3).
        Order: pitches_mean[12], pitches_std[12], pitches_min[12], pitches_max[12],
               timbre_mean[12],  timbre_std[12],  timbre_min[12],  timbre_max[12],
               loudness_mean, loudness_std, loudness_max_agg.

    """
    result: list[float] = []
    for arr in (pitches, timbre):
        if arr.size == 0:
            result.extend([0.0] * 48)
            continue
        result.extend(float(x) for x in np.mean(arr, axis=0))
        result.extend(float(x) for x in np.std(arr, axis=0, ddof=0))
        result.extend(float(x) for x in np.min(arr, axis=0))
        result.extend(float(x) for x in np.max(arr, axis=0))
    if loudness_max.size == 0:
        result.extend([0.0, 0.0, 0.0])
    else:
        result.append(float(np.mean(loudness_max)))
        result.append(float(np.std(loudness_max, ddof=0)))
        result.append(float(np.max(loudness_max)))
    return np.array(result, dtype=np.float64)
