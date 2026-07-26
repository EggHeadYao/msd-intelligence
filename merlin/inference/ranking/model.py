"""Logistic Ranker scoring, artifact publication, and lineage loading."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..artifacts.integrity import sha256_path
from ..artifacts.io import write_json_atomic
from .features import FEATURE_ORDER, FEATURE_SCHEMA


RANKER_TRAINING_VERSION = "merlin_ranker_training_v1"


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


def write_ranker_artifacts(
    output_dir: str | Path,
    *,
    fill_values: Mapping[str, float],
    means: Sequence[float],
    stds: Sequence[float],
    coefficients: Sequence[float],
    intercept: float,
    reg_param: float,
    stage: str,
    converged: bool,
    iterations: int,
    selection: Mapping[str, object],
    parent_paths: Mapping[str, str | Path],
    scope: str,
    constant_features: Sequence[str] = (),
) -> dict[str, object]:
    if scope not in {"formal", "smoke"}:
        raise ValueError("ranker scope must be formal or smoke")
    size = len(FEATURE_ORDER)
    if not (len(means) == len(stds) == len(coefficients) == size):
        raise ValueError("ranker artifact vector length mismatch")
    if any(float(value) <= 0.0 for value in stds):
        raise ValueError("ranker scaler standard deviations must be positive")
    constant = tuple(str(name) for name in constant_features)
    if len(set(constant)) != len(constant) or any(
        name not in FEATURE_ORDER for name in constant
    ):
        raise ValueError("ranker constant-feature list is invalid")
    if not converged:
        raise ValueError("ranker run did not converge")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    schema_path = root / "ranker_feature_schema.json"
    scaler_path = root / "ranker_scaler.json"
    coefficients_path = root / "ranker_coefficients.json"
    manifest_path = root / "training_manifest.json"
    common = {
        "feature_schema_version": FEATURE_SCHEMA,
        "feature_order": list(FEATURE_ORDER),
    }
    write_json_atomic(
        {
            **common,
            "artifact_type": "ranker_feature_schema",
            "schema_version": 1,
        },
        schema_path,
    )
    write_json_atomic(
        {
            **common,
            "artifact_type": "ranker_scaler",
            "fit_split": "set_a",
            "training_universe": "set_a" if stage == "tuning" else "a_b_remaining",
            "fill_values": {name: float(value) for name, value in fill_values.items()},
            "means": [float(value) for value in means],
            "stds": [float(value) for value in stds],
            "constant_features": list(constant),
            "constant_feature_scale": "effective_std_1_with_zero_model_weight",
        },
        scaler_path,
    )
    write_json_atomic(
        {
            **common,
            "artifact_type": "ranker_coefficients",
            "model_type": "logistic_regression",
            "model_version": "full-merlin-lr-v1",
            "elastic_net_param": 0.0,
            "reg_param": float(reg_param),
            "max_iter": 100,
            "tol": 1e-6,
            "fit_intercept": True,
            "standardization": True,
            "class_weight": "none",
            "coefficients": [float(value) for value in coefficients],
            "intercept": float(intercept),
        },
        coefficients_path,
    )
    manifest = {
        **common,
        "artifact_type": "ranker_training",
        "artifact_version": RANKER_TRAINING_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "stage": stage,
        "selected_reg_param": float(reg_param),
        "class_weight": "none",
        "converged": converged,
        "iterations": int(iterations),
        "selection": dict(selection),
        "constant_features": list(constant),
        "artifact_hashes": {
            path.name: sha256_path(path)
            for path in (schema_path, scaler_path, coefficients_path)
        },
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_ranker_bundle(
    schema_path: str | Path,
    scaler_path: str | Path,
    coefficients_path: str | Path,
    training_manifest_path: str | Path,
    *,
    expected_parent_hashes: Mapping[str, str],
    expected_scope: str = "formal",
) -> LogisticRanker:
    """Validate one inseparable Ranker bundle and return its scorer."""
    paths = tuple(Path(path) for path in (schema_path, scaler_path, coefficients_path))
    manifest_path = Path(training_manifest_path)
    for path in (*paths, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"ranker artifact does not exist: {path}")
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != "ranker_training":
        raise ValueError("Ranker training manifest artifact type mismatch")
    if manifest.get("artifact_version") != RANKER_TRAINING_VERSION:
        raise ValueError("Ranker training manifest version mismatch")
    if manifest.get("scope") != expected_scope:
        raise ValueError("Ranker training manifest scope mismatch")
    if manifest.get("converged") is not True:
        raise ValueError("Ranker training manifest is not converged")
    if manifest.get("class_weight") != "none":
        raise ValueError("Ranker training manifest must disable class weights")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA:
        raise ValueError("Ranker training manifest schema version mismatch")
    if tuple(manifest.get("feature_order", ())) != FEATURE_ORDER:
        raise ValueError("Ranker training manifest feature order mismatch")

    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
def _floats(values: Sequence[object]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _read_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)
