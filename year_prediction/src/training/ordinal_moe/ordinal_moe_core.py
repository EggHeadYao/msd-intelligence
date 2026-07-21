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
