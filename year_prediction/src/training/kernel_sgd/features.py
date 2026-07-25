from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
from pyspark import cloudpickle

cloudpickle.register_pickle_by_value(sys.modules[__name__])

Point = tuple[str, str, int, float, np.ndarray]
PARTITION_BATCH_SIZE = 512


@dataclass(frozen=True)
class Transform:
    kind: str
    input_dimension: int
    output_dimension: int
    matrix: np.ndarray
    offset: np.ndarray


def pca_transform(projection: np.ndarray) -> Transform:
    values = np.asarray(projection, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("PCA projection must be a finite matrix")
    return Transform("pca", values.shape[0], values.shape[1], values, np.empty(0))


def rff_transform(input_dimension: int, output_dimension: int, gamma: float, seed: int) -> Transform:
    if min(input_dimension, output_dimension) <= 0 or not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("RFF dimensions and gamma must be positive")
    random = np.random.default_rng(seed)
    frequencies = random.normal(
        0.0, math.sqrt(2.0 * gamma), size=(output_dimension, input_dimension)
    )
    offsets = random.uniform(0.0, 2.0 * math.pi, size=output_dimension)
    return Transform("rff", input_dimension, output_dimension, frequencies, offsets)


def apply_transform(values: np.ndarray, transform: Transform, concatenate: bool = False) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (transform.input_dimension,):
        raise ValueError("feature dimension differs from transform")
    if transform.kind == "pca":
        mapped = vector @ transform.matrix
    elif transform.kind == "rff":
        scale = math.sqrt(2.0 / transform.output_dimension)
        mapped = scale * np.cos(transform.matrix @ vector + transform.offset)
    else:
        raise ValueError("unsupported transform kind")
    return np.concatenate((vector, mapped)) if concatenate else mapped


def transform_partition(
    rows: Iterable[Point], transform: Transform, concatenate: bool = False,
    batch_size: int = PARTITION_BATCH_SIZE,
) -> Iterator[Point]:
    for batch in point_batches(rows, batch_size):
        yield from _transform_batch(batch, transform, concatenate)


def _transform_batch(
    rows: list[Point], transform: Transform, concatenate: bool
) -> Iterator[Point]:
    values = np.stack([row[4] for row in rows])
    if values.shape[1] != transform.input_dimension:
        raise ValueError("feature dimension differs from transform")
    if transform.kind == "pca":
        mapped = values @ transform.matrix
    elif transform.kind == "rff":
        scale = math.sqrt(2.0 / transform.output_dimension)
        mapped = scale * np.cos(values @ transform.matrix.T + transform.offset)
    else:
        raise ValueError("unsupported transform kind")
    output = np.concatenate((values, mapped), axis=1) if concatenate else mapped
    for row, transformed in zip(rows, output):
        yield row[0], row[1], row[2], row[3], transformed


def point_batches(
    rows: Iterable[Point], batch_size: int = PARTITION_BATCH_SIZE
) -> Iterator[list[Point]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
