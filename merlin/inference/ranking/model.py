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
    def from_artifacts(
        cls,
        schema_path: str | Path,
        scaler_path: str | Path,
        coefficients_path: str | Path,
    ) -> "LogisticRanker":
        """Load the formal split schema, scaler, and LR coefficient artifacts."""
        schema = _read_json(schema_path)
        scaler = _read_json(scaler_path)
        model = _read_json(coefficients_path)
        version = str(schema["feature_schema_version"])
        order = tuple(schema["feature_order"])
        for name, artifact in (("scaler", scaler), ("coefficients", model)):
            if artifact.get("feature_schema_version") != version:
                raise ValueError(f"{name} artifact schema version mismatch")
            if tuple(artifact.get("feature_order", ())) != order:
                raise ValueError(f"{name} artifact feature order mismatch")
        if model.get("model_type") != "logistic_regression":
            raise ValueError("unsupported ranker model_type")
        return cls(
            feature_schema_version=version,
            feature_order=order,
            means=_floats(scaler["means"]),
            stds=_floats(scaler["stds"]),
            coefficients=_floats(model["coefficients"]),
            intercept=float(model["intercept"]),
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

def _floats(values: Sequence[object]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _read_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)
