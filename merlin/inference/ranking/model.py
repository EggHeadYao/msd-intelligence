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


RANKER_TRAINING_VERSION = "merlin_ranker_training_v4"
RANKING_LIMIT = 20
RELATION_GATE_POLICY = "c1_fallback_below_mean_relation_evidence_threshold"


def _query_signals(
    features: Sequence[Mapping[str, float]],
    *,
    include_audio: bool,
) -> tuple[tuple[float, ...], float]:
    if not features:
        return (), 0.0
    audio: list[float] = []
    bfs_total = 0.0
    tag_total = 0.0
    release_total = 0.0
    required = (
        "bfs_score",
        "has_bfs",
        "tag_tfidf_cosine",
        "has_tags",
        "same_release",
    )
    for row in features:
        missing = [name for name in required if name not in row]
        if include_audio and "cos_audio" not in row:
            missing.append("cos_audio")
        if missing:
            raise ValueError(f"ranker features missing: {missing}")
        values = tuple(float(row[name]) for name in required)
        if include_audio:
            audio_value = float(row["cos_audio"])
            if not math.isfinite(audio_value):
                raise ValueError("ranker feature cos_audio is not finite")
            audio.append(audio_value)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("query relation evidence must be finite")
        bfs_total += max(0.0, values[0]) if values[1] > 0.0 else 0.0
        tag_total += max(0.0, values[2]) if values[3] > 0.0 else 0.0
        release_total += max(0.0, values[4])
    count = len(features)
    evidence = max(bfs_total, tag_total, release_total) / count
    return tuple(audio), evidence


def query_relation_evidence(
    features: Sequence[Mapping[str, float]],
) -> float:
    """Return mean relation-signal density for one recalled candidate list."""
    return _query_signals(features, include_audio=False)[1]


def quota_interleave_indices(
    candidate_ids: Sequence[str],
    learned_scores: Sequence[float],
    audio_scores: Sequence[float],
    audio_quota: int,
    *,
    limit: int = RANKING_LIMIT,
) -> tuple[int, ...]:
    """Merge learned and C1 orders with a deterministic Top-20 quota."""
    size = len(candidate_ids)
    if len(learned_scores) != size or len(audio_scores) != size:
        raise ValueError("ranking vector lengths differ")
    if len(set(candidate_ids)) != size:
        raise ValueError("ranking candidate IDs must be unique")
    if not 0 <= audio_quota <= RANKING_LIMIT:
        raise ValueError("audio quota must be between 0 and 20")
    if not 1 <= limit <= RANKING_LIMIT:
        raise ValueError("ranking limit must be between 1 and 20")
    if any(
        not math.isfinite(float(value))
        for value in (*learned_scores, *audio_scores)
    ):
        raise ValueError("ranking scores must be finite")
    if audio_quota == RANKING_LIMIT:
        return tuple(sorted(
            range(size),
            key=lambda index: (-audio_scores[index], candidate_ids[index]),
        )[:limit])
    if audio_quota == 0:
        return tuple(sorted(
            range(size),
            key=lambda index: (-learned_scores[index], candidate_ids[index]),
        )[:limit])
    learned_order = tuple(sorted(
        range(size), key=lambda index: (-learned_scores[index], candidate_ids[index])
    ))
    audio_order = tuple(sorted(
        range(size), key=lambda index: (-audio_scores[index], candidate_ids[index])
    ))
    return quota_interleave_ordered_indices(
        learned_order,
        audio_order,
        audio_quota,
        limit=limit,
    )


def quota_interleave_ordered_indices(
    learned_order: Sequence[int],
    audio_order: Sequence[int],
    audio_quota: int,
    *,
    limit: int = RANKING_LIMIT,
) -> tuple[int, ...]:
    """Merge pre-sorted learned and Audio orders without sorting again."""
    size = len(learned_order)
    if len(audio_order) != size:
        raise ValueError("ranking order lengths differ")
    if not 0 <= audio_quota <= RANKING_LIMIT:
        raise ValueError("audio quota must be between 0 and 20")
    if not 1 <= limit <= RANKING_LIMIT:
        raise ValueError("ranking limit must be between 1 and 20")
    if audio_quota == RANKING_LIMIT:
        return tuple(int(index) for index in audio_order[:limit])
    if audio_quota == 0:
        return tuple(int(index) for index in learned_order[:limit])
    orders = (learned_order, audio_order)
    pointers = [0, 0]
    selected: list[int] = []
    used: set[int] = set()
    for rank in range(1, min(limit, size) + 1):
        audio_due = (
            rank * audio_quota // RANKING_LIMIT
            > (rank - 1) * audio_quota // RANKING_LIMIT
        )
        source = int(audio_due)
        while orders[source][pointers[source]] in used:
            pointers[source] += 1
        index = orders[source][pointers[source]]
        pointers[source] += 1
        selected.append(index)
        used.add(index)
    return tuple(selected)


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
        raise ValueError("Ranker training manifest is missing artifact hashes")
    for path in paths:
        if artifact_hashes.get(path.name) != sha256_path(path):
            raise ValueError(f"Ranker artifact hash mismatch: {path.name}")

    parent_hashes = manifest.get("parent_hashes")
    if not isinstance(parent_hashes, dict):
        raise ValueError("Ranker training manifest is missing parent hashes")
    for name, expected in expected_parent_hashes.items():
        if parent_hashes.get(name) != expected:
            raise ValueError(f"Ranker parent hash mismatch: {name}")

    ranker = LogisticRanker.from_artifacts(*paths)
    if ranker.feature_schema_version != FEATURE_SCHEMA:
        raise ValueError("Ranker scorer schema version mismatch")
    if ranker.feature_order != FEATURE_ORDER:
        raise ValueError("Ranker scorer feature order mismatch")
    return ranker


def _floats(values: Sequence[object]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _read_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)
