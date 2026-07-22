from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
from pyspark import cloudpickle

cloudpickle.register_pickle_by_value(sys.modules[__name__])

Point = tuple[str, str, int, float, np.ndarray]


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
    rows: Iterable[Point], transform: Transform, concatenate: bool = False
) -> Iterator[Point]:
    for track, artist, year, label, values in rows:
        yield track, artist, year, label, apply_transform(values, transform, concatenate)
