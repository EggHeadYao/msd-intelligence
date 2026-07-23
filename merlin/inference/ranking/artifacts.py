"""Write the inseparable Ranker-v2 schema/scaler/model bundle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from ..artifact_lineage import sha256_path
from ..feature_schema import RANKER_V2_FEATURES, RANKER_V2_SCHEMA_VERSION
from ..jsonl_artifact import write_json_atomic


RANKER_TRAINING_VERSION = "merlin_ranker_training_v1"


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
) -> dict[str, object]:
    if scope not in {"formal", "smoke"}:
        raise ValueError("ranker scope must be formal or smoke")
    size = len(RANKER_V2_FEATURES)
    if not (len(means) == len(stds) == len(coefficients) == size):
        raise ValueError("ranker artifact vector length mismatch")
    if any(float(value) <= 0.0 for value in stds):
        raise ValueError("ranker scaler standard deviations must be positive")
    if not converged:
        raise ValueError("ranker run did not converge")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    schema_path = root / "ranker_feature_schema.json"
    scaler_path = root / "ranker_scaler.json"
    coefficients_path = root / "ranker_coefficients.json"
    manifest_path = root / "training_manifest.json"
    common = {
        "feature_schema_version": RANKER_V2_SCHEMA_VERSION,
        "feature_order": list(RANKER_V2_FEATURES),
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
            "fit_split": "set_a" if stage == "tuning" else "a_b_remaining",
            "fill_values": {name: float(value) for name, value in fill_values.items()},
            "means": [float(value) for value in means],
            "stds": [float(value) for value in stds],
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
        "converged": converged,
        "iterations": int(iterations),
        "selection": dict(selection),
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
