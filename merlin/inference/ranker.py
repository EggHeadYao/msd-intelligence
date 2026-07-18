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

    @classmethod
    def mock(cls, feature_schema_version: str, feature_order: Sequence[str]) -> "LogisticRanker":
        """Create an equal-weight artifact for pipeline integration tests."""
        size = len(feature_order)
        return cls(
            feature_schema_version=feature_schema_version,
            feature_order=tuple(feature_order),
            means=(0.0,) * size,
            stds=(1.0,) * size,
            coefficients=(1.0,) * size,
            intercept=0.0,
        )

    def score(self, features: Mapping[str, float]) -> float:
        """Return the raw LR margin used for ranking."""
        return self.raw_margin(features)

    def raw_margin(self, features: Mapping[str, float]) -> float:
        """Compute ``w*x+b`` after applying the frozen feature scaler."""
        missing = [name for name in self.feature_order if name not in features]
        if missing:
            raise ValueError(f"ranker features missing: {missing}")
        logit = self.intercept
        for name, mean, std, coefficient in zip(
            self.feature_order, self.means, self.stds, self.coefficients, strict=True
        ):
            value = float(features[name])
            if not math.isfinite(value):
                raise ValueError(f"ranker feature {name} is not finite")
            logit += coefficient * ((value - mean) / std)
        return logit

    def display_score(self, features: Mapping[str, float]) -> float:
        """Return a sigmoid display score; it is not a calibrated probability."""
        margin = self.raw_margin(features)
        if margin >= 0.0:
            return 1.0 / (1.0 + math.exp(-margin))
        exp_margin = math.exp(margin)
        return exp_margin / (1.0 + exp_margin)


def _floats(values: Sequence[object]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)
