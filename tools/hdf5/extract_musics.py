"""Extract segment arrays, similar artists, and artist terms from per-song .h5 files."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import h5py
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


def process_one_file(
    path: Path,
) -> tuple[np.ndarray, list[tuple[str, str]], list[tuple[str, str]]]:
    """Extract aggregated features, similar artists, and terms from one .h5 file.

    Args:
        path: Path to a per-song HDF5 file.

    Returns:
        Tuple of (features_99, similar_pairs, term_pairs).
        features_99: 1-D float64 numpy array of 99 aggregated segment features.
        similar_pairs: List of (track_id, similar_artist_id) with empty strings
                       filtered out.
        term_pairs: List of (track_id, term) with empty strings filtered out.

    """
    with h5py.File(path, "r") as h5:
        track_id: str = _decode(h5["/analysis/songs"][0]["track_id"])

        features: np.ndarray = aggregate_segments(
            h5["/analysis/segments_pitches"][:],
            h5["/analysis/segments_timbre"][:],
            h5["/analysis/segments_loudness_max"][:],
        )

        raw_similar: np.ndarray = h5["/metadata/similar_artists"][:]
        similar_pairs: list[tuple[str, str]] = [
            (track_id, _decode(aid)) for aid in raw_similar if _decode(aid)
        ]

        raw_terms: np.ndarray = h5["/metadata/artist_terms"][:]
        term_pairs: list[tuple[str, str]] = [
            (track_id, _decode(t)) for t in raw_terms if _decode(t)
        ]

    return features, similar_pairs, term_pairs


def _decode(val: object) -> str:
    """Decode HDF5 byte strings to Python str, pass through native str."""
    if isinstance(val, bytes | bytearray):
        return bytes(val).decode("utf-8")
    return str(val)
