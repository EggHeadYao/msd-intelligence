from __future__ import annotations

import sys

from pyspark import cloudpickle

cloudpickle.register_pickle_by_value(sys.modules[__name__])

import math
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Iterator, Sequence

import numpy as np
from pyspark import RDD

MIN_YEAR = 1922.0
MAX_YEAR = 2011.0
YEAR_SPAN = MAX_YEAR - MIN_YEAR
THRESHOLD_COUNT = int(YEAR_SPAN)
DECADE_COUNT = 10
DECADE_CENTERS = np.asarray(
    [1925.0, 1935.0, 1945.0, 1955.0, 1965.0, 1975.0, 1985.0, 1995.0, 2005.0, 2010.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class LossConfig:
    ordinal: float = 0.35
    moe: float = 0.45
    direct: float = 0.05
    decade: float = 0.12
    consistency: float = 0.03
    huber_delta: float = 3.0
    expert_span: float = 8.0
    blend_ordinal: float = 0.20
    blend_moe: float = 0.80
    blend_direct: float = 0.0

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("loss configuration contains non-finite values")
        if min(self.ordinal, self.moe, self.direct, self.decade, self.consistency) < 0.0:
            raise ValueError("loss weights cannot be negative")
        if self.huber_delta <= 0.0 or self.expert_span <= 0.0:
            raise ValueError("Huber delta and expert span must be positive")
        blend_sum = self.blend_ordinal + self.blend_moe + self.blend_direct
        if abs(blend_sum - 1.0) > 1.0e-9:
            raise ValueError("inference blend weights must sum to one")


@dataclass(frozen=True)
class ParameterLayout:
    dimension: int

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("feature dimension must be positive")

    def slices(self) -> dict[str, slice]:
        d = self.dimension
        cursor = 0
        result: dict[str, slice] = {}
        for name, size in (
            ("ordinal_w", d),
            ("ordinal_b", 1),
            ("thresholds", THRESHOLD_COUNT),
            ("gate_w", DECADE_COUNT * d),
            ("gate_b", DECADE_COUNT),
            ("expert_w", DECADE_COUNT * d),
            ("expert_b", DECADE_COUNT),
            ("direct_w", d),
            ("direct_b", 1),
        ):
            result[name] = slice(cursor, cursor + size)
            cursor += size
        return result

    @property
    def size(self) -> int:
        return self.slices()["direct_b"].stop

    def weight_mask(self) -> np.ndarray:
        mask = np.zeros(self.size, dtype=bool)
        slices = self.slices()
        for name in ("ordinal_w", "gate_w", "expert_w", "direct_w"):
            mask[slices[name]] = True
        return mask


def decade_index(years: np.ndarray) -> np.ndarray:
    return np.clip(((years.astype(np.int64) - 1920) // 10), 0, DECADE_COUNT - 1)


def _logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(probability, 1.0e-4, 1.0 - 1.0e-4)
    return np.log(values / (1.0 - values))


def initialize_parameters(
    layout: ParameterLayout,
    year_counts: Sequence[int],
    seed: int,
) -> np.ndarray:
    if len(year_counts) != THRESHOLD_COUNT + 1:
        raise ValueError("year histogram has the wrong dimension")
    rng = np.random.default_rng(seed)
    parameters = np.zeros(layout.size, dtype=np.float64)
    slices = layout.slices()
    total = max(1, int(sum(year_counts)))
    survival = np.asarray(
        [sum(year_counts[index + 1 :]) / total for index in range(THRESHOLD_COUNT)],
        dtype=np.float64,
    )
    parameters[slices["thresholds"]] = -_logit(survival)
    scale = 0.01 / math.sqrt(layout.dimension)
    for name in ("ordinal_w", "gate_w", "expert_w", "direct_w"):
        parameters[slices[name]] = rng.normal(0.0, scale, slices[name].stop - slices[name].start)
    return parameters


def unpack(parameters: np.ndarray, layout: ParameterLayout) -> dict[str, np.ndarray | float]:
    if parameters.shape != (layout.size,):
        raise ValueError("parameter vector has the wrong shape")
    d = layout.dimension
    slices = layout.slices()
    return {
        "ordinal_w": parameters[slices["ordinal_w"]],
        "ordinal_b": float(parameters[slices["ordinal_b"]][0]),
        "thresholds": parameters[slices["thresholds"]],
        "gate_w": parameters[slices["gate_w"]].reshape(DECADE_COUNT, d),
        "gate_b": parameters[slices["gate_b"]],
        "expert_w": parameters[slices["expert_w"]].reshape(DECADE_COUNT, d),
        "expert_b": parameters[slices["expert_b"]],
        "direct_w": parameters[slices["direct_w"]],
        "direct_b": float(parameters[slices["direct_b"]][0]),
    }

