from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def validate_source_mapping(
    index: Any,
    queries: Sequence[tuple[int, str, Sequence[float]]],
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, float]:
    if not queries:
        raise AssertionError("no query embeddings found")
    matrix = np.vstack(
        [np.asarray(row[2], dtype=np.float32).reshape(1, -1) for row in queries]
    )
    if matrix.shape[1] != index.d:
        raise AssertionError("query embedding dimension does not match FAISS index")
    if not np.all(np.isfinite(matrix)):
        raise AssertionError("query embedding contains NaN or infinite values")
    if not np.all(np.abs(np.linalg.norm(matrix, axis=1) - 1.0) <= tolerance):
        raise AssertionError("query embedding is not unit normalized")
    reconstructed = np.vstack([index.reconstruct(row[0]) for row in queries])
    error = float(np.max(np.abs(matrix - reconstructed)))
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError("FAISS row mapping does not match source embeddings")
    return matrix, error
