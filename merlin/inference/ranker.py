"""Versioned JSON logistic-ranker artifact and local scorer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class LogisticRanker:
    feature_schema_version: str
    feature_order: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        size = len(self.feature_order)
        if size == 0 or len(set(self.feature_order)) != size:
            raise ValueError("feature_order must be non-empty and unique")
        if not (len(self.means) == len(self.stds) == len(self.coefficients) == size):
            raise ValueError("ranker artifact vector lengths do not match feature_order")
        if any(not math.isfinite(value) for value in (*self.means, *self.stds, *self.coefficients, self.intercept)):
            raise ValueError("ranker artifact contains a non-finite number")
        if any(value <= 0.0 for value in self.stds):
            raise ValueError("ranker standard deviations must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> "LogisticRanker":
        with Path(path).open("r", encoding="utf-8") as stream:
            artifact = json.load(stream)
        if artifact.get("model_type") != "logistic_regression":
            raise ValueError("unsupported ranker model_type")
        return cls(
            feature_schema_version=str(artifact["feature_schema_version"]),
            feature_order=tuple(artifact["feature_order"]),
            means=_floats(artifact["means"]),
            stds=_floats(artifact["stds"]),
            coefficients=_floats(artifact["coefficients"]),
            intercept=float(artifact["intercept"]),
        )

def _floats(values: Sequence[object]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)
