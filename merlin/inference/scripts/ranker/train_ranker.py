"""Train the frozen Spark LR-L2 Ranker and publish Python inference artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from ...artifacts.integrity import sha256_path
from ...artifacts.paths import InferenceArtifactPaths
from ...ranking.model import write_ranker_artifacts
from ...ranking.features import (
    FEATURE_ORDER,
    FEATURE_SCHEMA,
    FILL_FEATURES,
    RAW_BASE_FEATURES,
    SAMPLE_WEIGHT_COLUMN,
    load_raw_feature_manifest,
)
from ...ranking.selection import (
    ADAPTIVE_AUDIO_QUOTAS,
    AUDIO_QUOTAS,
    REG_PARAMS,
    select_guarded_ranker_means,
)
from ..support.scratch import estimate_ranker_scratch_gb, prepare_scratch_root


QUERY_GROUPS = ("audio_dominant", "relation_dominant", "mixed")
RDD_COMPRESSION_CODEC = "lz4"
# Formal feature blocks measured 0.46; retain headroom for distribution changes.
RDD_STORAGE_RATIO = 0.60


@dataclass(frozen=True, slots=True)
class ValidationScoreTable:
    query_ids: tuple[str, ...]
    query_groups: tuple[str, ...]
    selection_folds: tuple[str, ...]
    relation_evidence: tuple[float, ...]
    columns: Mapping[object, Sequence[float]]

    def column(self, key: object) -> Sequence[float]:
        values = self.columns[key]
        if len(values) != len(self.query_ids):
            raise ValueError("Set-B score column length mismatch")
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-features-manifest", type=Path, required=True)
    parser.add_argument("--training-pairs-manifest", type=Path)
    parser.add_argument("--base-train-features", type=Path)
    parser.add_argument("--base-train-features-manifest", type=Path)
    parser.add_argument("--validation-features", type=Path)
    parser.add_argument("--validation-features-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("tuning", "final_retrain"), default="tuning")
    parser.add_argument(
        "--training-variant",
        choices=("full", "no_hard_neg"),
        default="full",
    )
    parser.add_argument("--fixed-reg-param", type=float)
    parser.add_argument("--frozen-scaler", type=Path)
    parser.add_argument("--frozen-tuning-manifest", type=Path)
    parser.add_argument("--parent", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--min-free-gb", type=float)
    parser.add_argument("--max-block-size-mb", type=float, default=32.0)
    return parser.parse_args()


def parse_parents(values: list[str]) -> dict[str, Path]:
    parents = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in parents:
            raise ValueError(f"invalid or duplicate parent binding: {value!r}")
        parents[name] = Path(path)
    return parents


def load_frozen_preprocessing(
    scaler_path: Path,
    tuning_manifest_path: Path,
    fixed_reg_param: float,
) -> tuple[
    dict[str, float],
    tuple[float, ...],
    tuple[float, ...],
    tuple[str, ...],
    int,
    float,
    int,
    float,
    bool,
]:
    with scaler_path.open("r", encoding="utf-8") as stream:
        scaler = json.load(stream)
    with tuning_manifest_path.open("r", encoding="utf-8") as stream:
        tuning = json.load(stream)
    if scaler.get("feature_schema_version") != FEATURE_SCHEMA:
        raise ValueError("frozen scaler schema version mismatch")
    if tuple(scaler.get("feature_order", ())) != FEATURE_ORDER:
        raise ValueError("frozen scaler feature order mismatch")
    if tuning.get("artifact_type") != "ranker_training" or tuning.get("stage") != "tuning":
        raise ValueError("frozen tuning manifest is invalid")
    if tuning.get("class_weight") != "none":
        raise ValueError("frozen tuning model must not use class weights")
    if float(tuning.get("selected_reg_param", -1.0)) != fixed_reg_param:
        raise ValueError("fixed regParam does not match the tuning manifest")
    fill_values = {
        name: float(value) for name, value in scaler.get("fill_values", {}).items()
    }
    if set(fill_values) != set(FILL_FEATURES):
        raise ValueError("frozen scaler fill-value set mismatch")
    means = tuple(float(value) for value in scaler.get("means", ()))
    stds = tuple(float(value) for value in scaler.get("stds", ()))
    if len(means) != len(FEATURE_ORDER) or len(stds) != len(means):
        raise ValueError("frozen scaler vector length mismatch")
    if any(not math.isfinite(value) for value in (*fill_values.values(), *means, *stds)):
        raise ValueError("frozen scaler contains a non-finite value")
    if any(value <= 0.0 for value in stds):
        raise ValueError("frozen scaler standard deviations must be positive")
    constant_features = tuple(str(name) for name in scaler.get("constant_features", ()))
    if tuple(tuning.get("constant_features", ())) != constant_features:
        raise ValueError("frozen constant-feature contract mismatch")
    selection = tuning.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("frozen tuning manifest is missing selection metadata")
    audio_quota = int(selection.get("selected_audio_quota", -1))
    if audio_quota not in AUDIO_QUOTAS:
        raise ValueError("frozen tuning audio quota is outside the selection grid")
    gate_threshold = float(
        selection.get("selected_relation_gate_threshold", -1.0)
    )
    if not math.isfinite(gate_threshold) or gate_threshold < 0.0:
        raise ValueError("frozen tuning relation gate threshold is invalid")
    high_audio_quota = int(
        selection.get("selected_high_evidence_audio_quota", audio_quota)
    )
    high_gate_threshold = float(
        selection.get("selected_high_relation_gate_threshold", gate_threshold)
    )
    if high_audio_quota not in AUDIO_QUOTAS or high_audio_quota > audio_quota:
        raise ValueError("frozen high-evidence Audio quota is invalid")
    if not math.isfinite(high_gate_threshold) or high_gate_threshold < gate_threshold:
        raise ValueError("frozen high relation gate threshold is invalid")
    publishable_fusion = selection.get("publishable_fusion") is True
    return (
        fill_values,
        means,
        stds,
        constant_features,
        audio_quota,
        gate_threshold,
        high_audio_quota,
        high_gate_threshold,
        publishable_fusion,
    )


def load_frozen_initial_parameters(
    tuning_manifest_path: Path,
    fixed_reg_param: float,
) -> tuple[Path, tuple[float, ...], float]:
    """Load the lineage-bound tuning solution used to warm-start LR."""
    with tuning_manifest_path.open("r", encoding="utf-8") as stream:
        tuning = json.load(stream)
    coefficients_path = tuning_manifest_path.parent / "ranker_coefficients.json"
    artifact_hashes = tuning.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or artifact_hashes.get(
        coefficients_path.name
    ) != sha256_path(coefficients_path):
        raise ValueError("frozen tuning coefficient hash mismatch")
    with coefficients_path.open("r", encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("artifact_type") != "ranker_coefficients":
        raise ValueError("frozen coefficient artifact type mismatch")
    if artifact.get("class_weight") != "none":
        raise ValueError("frozen coefficient model must not use class weights")
    if artifact.get("feature_schema_version") != FEATURE_SCHEMA:
        raise ValueError("frozen coefficient schema version mismatch")
    if tuple(artifact.get("feature_order", ())) != FEATURE_ORDER:
        raise ValueError("frozen coefficient feature order mismatch")
    if float(artifact.get("reg_param", -1.0)) != fixed_reg_param:
        raise ValueError("frozen coefficient regParam mismatch")
    coefficients = tuple(float(value) for value in artifact.get("coefficients", ()))
    intercept = float(artifact.get("intercept", float("nan")))
    if len(coefficients) != len(FEATURE_ORDER):
        raise ValueError("frozen coefficient vector length mismatch")
    if any(not math.isfinite(value) for value in (*coefficients, intercept)):
        raise ValueError("frozen coefficient artifact contains a non-finite value")
    return coefficients_path, coefficients, intercept


def set_initial_lr_model(
    estimator: Any,
    spark: Any,
    coefficients: tuple[float, ...],
    intercept: float,
) -> None:
    """Attach a binary Spark LR model as the optimizer's initial solution."""
    from pyspark.ml.common import _py2java
    from pyspark.ml.linalg import Vectors

    context = spark.sparkContext
    java_coefficients = _py2java(context, Vectors.dense(coefficients))
    java_model = context._jvm.org.apache.spark.ml.classification.LogisticRegressionModel(
        f"{estimator.uid}_initial",
        java_coefficients,
        float(intercept),
    )
    estimator._java_obj.setInitialModel(java_model)


def solver_feature_order(constant_features: tuple[str, ...]) -> tuple[str, ...]:
    """Return the fitted dimensions after enforcing frozen zero coefficients."""
    constant = set(constant_features)
    if len(constant) != len(constant_features) or not constant.issubset(
        FEATURE_ORDER
    ):
        raise ValueError("frozen constant-feature set is invalid")
    features = tuple(name for name in FEATURE_ORDER if name not in constant)
    if not features:
        raise ValueError("ranker must retain at least one fitted feature")
    return features


def expand_solver_coefficients(
    solver_features: tuple[str, ...],
    fitted_coefficients: tuple[float, ...],
) -> tuple[float, ...]:
    """Restore the published feature order with zeroes for omitted dimensions."""
    if len(solver_features) != len(fitted_coefficients):
        raise ValueError("solver coefficient vector length mismatch")
    if len(set(solver_features)) != len(solver_features) or any(
        name not in FEATURE_ORDER for name in solver_features
    ):
        raise ValueError("solver feature order is invalid")
    fitted_by_name = dict(zip(solver_features, fitted_coefficients, strict=True))
    return tuple(fitted_by_name.get(name, 0.0) for name in FEATURE_ORDER)


def _collect_validation_scores(frame: Any, models: Mapping) -> ValidationScoreTable:
    """Score every Set-B ranker and baseline in one query-grouped Spark job."""
    import numpy as np

    coefficient_rows = tuple(
        tuple(float(value) for value in models[reg_param].coefficients)
        for reg_param in REG_PARAMS
    )
    intercepts = tuple(float(models[reg_param].intercept) for reg_param in REG_PARAMS)
    configurations = tuple(
        (reg_param, quota)
        for reg_param in REG_PARAMS
        for quota in AUDIO_QUOTAS
    )
    model_names = tuple(
        f"reg:{reg_param:.17g}:audio_quota:{quota}"
        for reg_param, quota in configurations
    )
    config_by_scorer = dict(zip(model_names, configurations, strict=True))
    scorer_names = model_names + ("c1_only", "c2_only", "bfs")
    quota_sources = tuple(
        tuple(
            int(
                rank * quota // 20
                > (rank - 1) * quota // 20
            )
            for rank in range(1, 21)
        )
        for quota in AUDIO_QUOTAS
    )
    coefficient_matrix = np.asarray(coefficient_rows, dtype=np.float64).T
    intercept_array = np.asarray(intercepts, dtype=np.float64)
    discounts = np.asarray(
        tuple(1.0 / math.log2(rank + 1.0) for rank in range(1, 21)),
        dtype=np.float64,
    )

    def score_query(query):
        import numpy as np
        import pandas as pd

        query_id = str(query["query_track_id"].iloc[0])
        folds = set(query["selection_fold"].astype(str))
        if len(folds) != 1:
            raise ValueError("Set-B query crosses selection folds")
        selection_fold = next(iter(folds))
        candidate_ids = query["candidate_track_id"].astype(str).to_numpy()
        feature_matrix = np.vstack(query["feature_array"].to_numpy()).astype(
            np.float64, copy=False
        )
        lr_score_matrix = feature_matrix @ coefficient_matrix + intercept_array
        audio_scores = query["cos_audio"].to_numpy(dtype=np.float64)
        bfs_evidence = np.where(
            query["has_bfs"].to_numpy(dtype=np.float64) > 0.0,
            np.maximum(query["bfs_score"].to_numpy(dtype=np.float64), 0.0),
            0.0,
        )
        tag_evidence = np.where(
            query["has_tags"].to_numpy(dtype=np.float64) > 0.0,
            np.maximum(
                query["tag_tfidf_cosine"].to_numpy(dtype=np.float64), 0.0
            ),
            0.0,
        )
        release_evidence = np.maximum(
            query["same_release"].to_numpy(dtype=np.float64), 0.0
        )
        relation_evidence = float(max(
            np.mean(bfs_evidence),
            np.mean(tag_evidence),
            np.mean(release_evidence),
        ))
        baseline_scores = {
            "c1_only": audio_scores,
            "c2_only": np.where(
                query["has_graph"].to_numpy(dtype=np.float64) > 0.0,
                query["cos_graph"].to_numpy(dtype=np.float64),
                -np.inf,
            ),
            "bfs": np.where(
                query["has_bfs"].to_numpy(dtype=np.float64) > 0.0,
                query["bfs_score"].to_numpy(dtype=np.float64),
                -np.inf,
            ),
        }
        labels = {
            group: np.zeros(len(query), dtype=np.int8) for group in QUERY_GROUPS
        }
        positive_counts: dict[str, int] = {}
        for row_index, groups in enumerate(query["validation_groups"]):
            for group in groups:
                values = group.asDict() if hasattr(group, "asDict") else group
                name = str(values["query_group"])
                labels[name][row_index] = int(values["label"])
                positive_counts[name] = max(
                    positive_counts.get(name, 0),
                    int(values["eligible_positive_count"]),
                )

        def top_indices(values, limit=20):
            size = len(values)
            if size <= limit:
                return np.lexsort((candidate_ids, -values))
            cutoff = np.partition(values, size - limit)[size - limit]
            better = np.flatnonzero(values > cutoff)
            tied = np.flatnonzero(values == cutoff)
            needed = limit - len(better)
            selected_ties = tied[
                np.argsort(candidate_ids[tied], kind="stable")[:needed]
            ]
            selected = np.concatenate((better, selected_ties))
            return selected[np.lexsort((candidate_ids[selected], -values[selected]))]

        audio_order = np.lexsort((candidate_ids, -audio_scores))
        learned_orders = tuple(
            np.lexsort((candidate_ids, -lr_score_matrix[:, model_index]))
            for model_index in range(len(REG_PARAMS))
        )

        ranking_limit = min(20, len(query))

        def quota_top(learned_order, quota):
            if quota == 0:
                return learned_order[:ranking_limit]
            if quota == 20:
                return audio_order[:ranking_limit]
            sources = quota_sources[quota]
            pointers = [0, 0]
            orders = (learned_order, audio_order)
            selected = []
            used = set()
            for source in sources[:ranking_limit]:
                while int(orders[source][pointers[source]]) in used:
                    pointers[source] += 1
                index = int(orders[source][pointers[source]])
                pointers[source] += 1
                selected.append(index)
                used.add(index)
            return np.asarray(selected, dtype=np.int64)

        denominators = {
            group: float(np.sum(discounts[: min(positive_count, 20)]))
            for group, positive_count in positive_counts.items()
        }
        rankings = [
            quota_top(learned_order, quota)
            for learned_order in learned_orders
            for quota in AUDIO_QUOTAS
        ]
        rankings.append(audio_order[:ranking_limit])
        rankings.extend(
            top_indices(baseline_scores[name]) for name in ("c2_only", "bfs")
        )
        metrics = {group: [] for group in positive_counts}
        for top in rankings:
            weighted = discounts[: len(top)]
            for group in positive_counts:
                dcg = float(np.sum(weighted * labels[group][top]))
                metrics[group].append(dcg / denominators[group])
        rows = [
            {
                "query_track_id": query_id,
                "query_group": group,
                "selection_fold": selection_fold,
                "relation_evidence": relation_evidence,
                "score_values": metrics[group],
            }
            for group in positive_counts
        ]
        return pd.DataFrame.from_records(rows)

    rows = (
        frame.groupBy("query_track_id")
        .applyInPandas(
            score_query,
            "query_track_id string, query_group string, selection_fold string, "
            "relation_evidence double, score_values array<double>",
        )
        .collect()
    )
    keys = tuple(config_by_scorer.get(name, name) for name in scorer_names)
    query_ids = []
    query_groups = []
    selection_folds = []
    relation_evidence = []
    score_rows = []
    for row in rows:
        values = row["score_values"]
        if len(values) != len(keys):
            raise ValueError("Set-B wide score vector length mismatch")
        query_ids.append(str(row["query_track_id"]))
        query_groups.append(str(row["query_group"]))
        selection_folds.append(str(row["selection_fold"]))
        relation_evidence.append(float(row["relation_evidence"]))
        score_rows.append(values)
    score_matrix = np.asarray(score_rows, dtype=np.float64)
    return ValidationScoreTable(
        query_ids=tuple(query_ids),
        query_groups=tuple(query_groups),
        selection_folds=tuple(selection_folds),
        relation_evidence=tuple(relation_evidence),
        columns={
            key: score_matrix[:, index]
            for index, key in enumerate(keys)
        },
    )


def _score_summary(by_group: Mapping[str, float]) -> dict[str, object]:
    return {
        "by_group": dict(by_group),
        "three_strata_macro": sum(by_group.values()) / len(QUERY_GROUPS),
    }


def _validate_score_rows(table: ValidationScoreTable) -> None:
    lengths = {
        len(table.query_ids),
        len(table.query_groups),
        len(table.selection_folds),
        len(table.relation_evidence),
    }
    if len(lengths) != 1:
        raise ValueError("Set-B score row lengths differ")
    seen = set()
    for query_id, group, fold in zip(
        table.query_ids,
        table.query_groups,
        table.selection_folds,
        strict=True,
    ):
        if group not in QUERY_GROUPS:
            raise ValueError(f"unknown Set-B query group: {group}")
        if fold not in {"tune", "confirm"}:
            raise ValueError(f"unknown Set-B selection fold: {fold}")
        key = (query_id, group)
        if key in seen:
            raise ValueError("duplicate Set-B query-group score")
        seen.add(key)


def _baseline_group_means(
    table: ValidationScoreTable,
    key: object,
    fold: str,
) -> dict[str, float]:
    import numpy as np

    values = np.asarray(table.column(key), dtype=np.float64)
    groups = np.asarray(table.query_groups, dtype=object)
    folds = np.asarray(table.selection_folds, dtype=object)
    means = {}
    for group in QUERY_GROUPS:
        selected = (folds == fold) & (groups == group)
        if not np.any(selected):
            raise ValueError(f"Set-B {fold} fold is missing a frozen query group")
        means[group] = float(np.mean(values[selected]))
    return means


def _configuration_group_means(
    table: ValidationScoreTable,
    thresholds: tuple[float, ...],
) -> dict[
    str,
    dict[tuple[float, int, int, float, float], dict[str, float]],
]:
    """Aggregate the full Set-B grid without retaining per-query dictionaries."""
    import numpy as np

    model_scores = {
        (reg_param, quota): np.asarray(
            table.column((reg_param, quota)), dtype=np.float64
        )
        for reg_param in REG_PARAMS
        for quota in ADAPTIVE_AUDIO_QUOTAS
    }
    c1 = np.asarray(table.column("c1_only"), dtype=np.float64)
    evidence = np.asarray(table.relation_evidence, dtype=np.float64)
    groups = np.asarray(table.query_groups, dtype=object)
    folds = np.asarray(table.selection_folds, dtype=object)
    grid = tuple(
        (reg_param, middle_quota, high_quota, low_threshold, high_threshold)
        for reg_param in REG_PARAMS
        for middle_quota in ADAPTIVE_AUDIO_QUOTAS
        for high_quota in ADAPTIVE_AUDIO_QUOTAS
        if high_quota <= middle_quota
        for low_index, low_threshold in enumerate(thresholds)
        for high_threshold in thresholds[low_index:]
    )
    aggregated = {
        fold: {configuration: {} for configuration in grid}
        for fold in ("tune", "confirm")
    }
    for fold in aggregated:
        for group in QUERY_GROUPS:
            selected = (folds == fold) & (groups == group)
            count = int(np.count_nonzero(selected))
            if count == 0:
                raise ValueError(f"Set-B {fold} fold is missing a frozen query group")
            for low_index, low_threshold in enumerate(thresholds):
                fallback = selected & (evidence < low_threshold)
                fallback_sum = float(np.sum(c1[fallback]))
                for high_threshold in thresholds[low_index:]:
                    middle = selected & (evidence >= low_threshold) & (
                        evidence < high_threshold
                    )
                    high = selected & (evidence >= high_threshold)
                    for reg_param in REG_PARAMS:
                        middle_sums = {
                            quota: float(np.sum(
                                model_scores[(reg_param, quota)][middle]
                            ))
                            for quota in ADAPTIVE_AUDIO_QUOTAS
                        }
                        high_sums = {
                            quota: float(np.sum(
                                model_scores[(reg_param, quota)][high]
                            ))
                            for quota in ADAPTIVE_AUDIO_QUOTAS
                        }
                        for middle_quota in ADAPTIVE_AUDIO_QUOTAS:
                            for high_quota in ADAPTIVE_AUDIO_QUOTAS:
                                if high_quota > middle_quota:
                                    continue
                                total = (
                                    fallback_sum
                                    + middle_sums[middle_quota]
                                    + high_sums[high_quota]
                                )
                                key = (
                                    reg_param,
                                    middle_quota,
                                    high_quota,
                                    low_threshold,
                                    high_threshold,
                                )
                                aggregated[fold][key][group] = total / count
    return aggregated


def _relation_gate_thresholds(
    validation_scores: ValidationScoreTable,
) -> tuple[float, ...]:
    import numpy as np

    by_query: dict[str, float] = {}
    for query_id, fold, evidence in zip(
        validation_scores.query_ids,
        validation_scores.selection_folds,
        validation_scores.relation_evidence,
        strict=True,
    ):
        if fold != "tune":
            continue
        previous = by_query.setdefault(query_id, evidence)
        if previous != evidence:
            raise ValueError("relation evidence changed across query groups")
    if not by_query:
        raise ValueError("Set-B tune fold has no relation evidence")
    values = np.asarray(tuple(by_query.values()), dtype=np.float64)
    thresholds = {
        float(value)
        for value in np.quantile(values, np.linspace(0.0, 1.0, 21))
    }
    thresholds.add(math.nextafter(float(values.max()), math.inf))
    return tuple(sorted(thresholds))


def _select_ranker_configuration(
    validation_scores: ValidationScoreTable,
) -> tuple[float, int, int, float, float, dict[str, object]]:
    """Tune an adaptive quota once and verify it on Set-B confirmation."""
    _validate_score_rows(validation_scores)
    thresholds = _relation_gate_thresholds(validation_scores)
    scores = _configuration_group_means(validation_scores, thresholds)
    selected_key, report = select_guarded_ranker_means(
        scores["tune"],
        scores["confirm"],
        _baseline_group_means(validation_scores, "c1_only", "tune"),
        _baseline_group_means(validation_scores, "c1_only", "confirm"),
    )
    (
        selected_reg,
        selected_quota,
        selected_high_quota,
        selected_threshold,
        selected_high_threshold,
    ) = selected_key
    report["set_b_diagnostics"] = {
        fold: {
            scorer: _score_summary(
                _baseline_group_means(validation_scores, scorer, fold)
            )
            for scorer in ("c1_only", "c2_only", "bfs")
        }
        for fold in ("tune", "confirm")
    }
    report["configuration_grid"] = {
        "reg_params": list(REG_PARAMS),
        "audio_quotas": list(ADAPTIVE_AUDIO_QUOTAS),
        "relation_gate_thresholds": list(thresholds),
    }
    return (
        selected_reg,
        selected_quota,
        selected_high_quota,
        selected_threshold,
        selected_high_threshold,
        report,
    )


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.max_block_size_mb) or args.max_block_size_mb <= 0.0:
        raise ValueError("max-block-size-mb must be a positive finite value")
    if args.stage == "tuning" and args.validation_features is None:
        raise ValueError("tuning requires validation features")
    if args.stage == "tuning" and args.validation_features_manifest is None:
        raise ValueError("tuning requires a validation feature manifest")
    if args.stage == "tuning" and args.training_variant != "full":
        raise ValueError("no-hard-neg is defined for retrain only")
    if args.stage == "final_retrain" and args.fixed_reg_param not in REG_PARAMS:
        raise ValueError("retrain requires one frozen regParam")
    if args.stage == "final_retrain" and (
        args.frozen_scaler is None or args.frozen_tuning_manifest is None
    ):
        raise ValueError("retrain requires the frozen Set-A scaler and tuning manifest")
    train_feature_manifest = load_raw_feature_manifest(
        args.train_features_manifest,
        args.train_features,
        expected_scope=args.scope,
        expected_pair_kind="training",
        expected_stage=args.stage,
    )
    base_feature_manifest = None
    base_features = None
    base_features_manifest = None
    if args.training_variant == "no_hard_neg":
        paths = InferenceArtifactPaths()
        base_features = args.base_train_features or paths.raw_pair_features
        base_features_manifest = (
            args.base_train_features_manifest or paths.raw_pair_features_manifest
        )
        base_feature_manifest = load_raw_feature_manifest(
            base_features_manifest,
            base_features,
            expected_scope=args.scope,
            expected_pair_kind="training",
            expected_stage="final_retrain",
        )
        pair_manifest_path = (
            args.training_pairs_manifest or paths.no_hard_neg_pairs_manifest
        )
        parents = train_feature_manifest.get("parent_hashes")
        if not isinstance(parents, dict) or parents.get(
            "training_pairs_manifest"
        ) != sha256_path(pair_manifest_path):
            raise ValueError("no-hard-neg features are not bound to its pair manifest")
        with pair_manifest_path.open("r", encoding="utf-8") as stream:
            no_hard_pairs = json.load(stream)
        if float(no_hard_pairs.get("candidate_aware_target_fraction", -1.0)) != 0.0:
            raise ValueError("no-hard-neg pair manifest contains hard negatives")
        if no_hard_pairs.get("dataset_layout") != "random_replacement_delta":
            raise ValueError("no-hard-neg artifact is not a replacement delta")
        if parents.get("full_raw_features_manifest") != sha256_path(
            base_features_manifest
        ):
            raise ValueError("no-hard-neg delta is not bound to its Full base")
        if int(train_feature_manifest.get("effective_row_count", -1)) != int(
            base_feature_manifest["row_count"]
        ):
            raise ValueError("no-hard-neg delta and Full base row counts differ")
    validation_feature_manifest = None
    frozen_preprocessing = None
    frozen_initial = None
    frozen_coefficients_path = None
    if args.stage == "tuning":
        validation_feature_manifest = load_raw_feature_manifest(
            args.validation_features_manifest,
            args.validation_features,
            expected_scope=args.scope,
            expected_pair_kind="validation",
            expected_stage="tuning",
        )
    else:
        frozen_preprocessing = load_frozen_preprocessing(
            args.frozen_scaler,
            args.frozen_tuning_manifest,
            float(args.fixed_reg_param),
        )
        frozen_coefficients_path, frozen_coefficients, frozen_intercept = (
            load_frozen_initial_parameters(
                args.frozen_tuning_manifest,
                float(args.fixed_reg_param),
            )
        )
        frozen_initial = (frozen_coefficients, frozen_intercept)
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from pyspark.ml.functions import array_to_vector, vector_to_array
    from pyspark import StorageLevel
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    constant_features = (
        frozen_preprocessing[3] if frozen_preprocessing is not None else ()
    )
    solver_features = solver_feature_order(constant_features)
    projected_gb = estimate_ranker_scratch_gb(
        training_rows=int(train_feature_manifest.get(
            "effective_row_count",
            train_feature_manifest["row_count"],
        )),
        validation_rows=(
            int(validation_feature_manifest["row_count"])
            if validation_feature_manifest is not None
            else 0
        ),
        feature_count=len(solver_features),
        model_count=len(REG_PARAMS) if args.stage == "tuning" else 1,
        training_storage_ratio=RDD_STORAGE_RATIO,
    )
    prepare_scratch_root(
        args.output.parent,
        scope=args.scope,
        min_free_gb=args.min_free_gb,
        projected_gb=projected_gb,
    )
    scratch_root = prepare_scratch_root(
        args.scratch_root or args.output.parent / ".c3-scratch",
        scope=args.scope,
        min_free_gb=args.min_free_gb,
        projected_gb=projected_gb,
    )
    spark_local_temporary = TemporaryDirectory(prefix="merlin-ranker-spark-", dir=scratch_root)
    spark = (
        SparkSession.builder.appName("MerlinRankerTraining")
        .config("spark.local.dir", spark_local_temporary.name)
        .config("spark.rdd.compress", "true")
        .config("spark.io.compression.codec", RDD_COMPRESSION_CODEC)
        .getOrCreate()
    )
    cached = []
    try:
        def release(frame: Any) -> None:
            frame.unpersist(blocking=True)
            cached[:] = [item for item in cached if item is not frame]

        def read_rows(path: Path):
            return (
                spark.read.parquet(str(path))
                if path.suffix == ".parquet"
                else spark.read.json(str(path))
            )

        train_columns = ("label", SAMPLE_WEIGHT_COLUMN, *RAW_BASE_FEATURES)
        train = read_rows(args.train_features).select(*train_columns)
        if args.training_variant == "no_hard_neg":
            assert base_features is not None
            base = read_rows(base_features).select(
                "label",
                SAMPLE_WEIGHT_COLUMN,
                "negative_source",
                *RAW_BASE_FEATURES,
            )
            base = base.where(
                F.col("negative_source").isNull()
                | (F.col("negative_source") != F.lit("candidate_aware"))
            ).withColumn(
                SAMPLE_WEIGHT_COLUMN,
                F.when(F.col("label") == 0, F.lit(1.0)).otherwise(
                    F.col(SAMPLE_WEIGHT_COLUMN)
                ),
            ).select(*train_columns)
            train = base.unionByName(train)
        if args.stage == "tuning":
            train = train.persist(StorageLevel.MEMORY_AND_DISK)
            cached.append(train)
            statistics = train.agg(
                F.count("*").alias("row_count"),
                *(
                    F.percentile_approx(name, 0.5, 1_000_000).alias(f"median_{name}")
                    for name in FILL_FEATURES
                ),
            ).first()
            if int(statistics["row_count"]) == 0:
                raise ValueError("Ranker training features are empty")
            fill_values = {}
            for name in FILL_FEATURES:
                value = statistics[f"median_{name}"]
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"Set-A fill statistic is invalid: {name}")
                fill_values[name] = float(value)
        else:
            (
                fill_values,
                means,
                stds,
                constant_features,
                _,
                _,
                _,
            ) = frozen_preprocessing

        def materialize(frame: Any) -> Any:
            result = frame
            for name, value in fill_values.items():
                result = result.withColumn(name, F.coalesce(F.col(name), F.lit(value)))
            result = result.withColumn(
                "audio_tag_interaction",
                F.col("cos_audio") * F.col("tag_tfidf_cosine"),
            ).withColumn(
                "graph_bfs_interaction",
                F.col("cos_graph") * F.col("bfs_score"),
            )
            return result

        assembler = VectorAssembler(
            inputCols=list(FEATURE_ORDER),
            outputCol="unscaled_features",
            handleInvalid="error",
        )
        materialized_train = materialize(train)
        if args.stage == "tuning":
            assembled_train = assembler.transform(materialized_train)
            scaler = StandardScaler(
                inputCol="unscaled_features",
                outputCol="features",
                withMean=True,
                withStd=True,
            ).fit(assembled_train)
            transformed_train = scaler.transform(assembled_train)
            means = tuple(float(value) for value in scaler.mean)
            spark_stds = tuple(float(value) for value in scaler.std)
            if any(not math.isfinite(value) or value < 0.0 for value in spark_stds):
                raise ValueError("Ranker scaler produced an invalid standard deviation")
            constant_features = tuple(
                name
                for name, value in zip(FEATURE_ORDER, spark_stds, strict=True)
                if value == 0.0
            )
            stds = tuple(1.0 if value == 0.0 else value for value in spark_stds)
        else:
            feature_indexes = {
                name: index for index, name in enumerate(FEATURE_ORDER)
            }
            scaled = F.array(*(
                (F.col(name) - F.lit(means[feature_indexes[name]]))
                / F.lit(stds[feature_indexes[name]])
                for name in solver_features
            ))
            transformed_train = materialized_train.select(
                F.col("label").cast("double").alias("label"),
                array_to_vector(scaled).alias("features"),
            )
        if args.stage == "tuning":
            scaled_train = transformed_train.select(
                F.col("label").cast("double").alias("label"),
                "features",
            ).persist(StorageLevel.MEMORY_AND_DISK)
            cached.append(scaled_train)
            scaled_train.count()
            release(train)
        else:
            scaled_train = transformed_train

        def fit(reg_param: float) -> Any:
            estimator = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                elasticNetParam=0.0,
                regParam=reg_param,
                maxIter=100,
                tol=1e-6,
                fitIntercept=True,
                standardization=True,
                family="binomial",
                maxBlockSizeInMB=args.max_block_size_mb,
            )
            if args.stage == "final_retrain":
                assert frozen_initial is not None
                initial_coefficients, initial_intercept = frozen_initial
                active_initial = tuple(
                    initial_coefficients[FEATURE_ORDER.index(name)]
                    for name in solver_features
                )
                set_initial_lr_model(
                    estimator,
                    spark,
                    active_initial,
                    initial_intercept,
                )
            model = estimator.fit(scaled_train)
            if int(model.summary.totalIterations) >= 100:
                raise ValueError(f"LR did not converge for regParam={reg_param}")
            return model

        selection: dict[str, object]
        if args.stage == "tuning":
            models = {reg_param: fit(reg_param) for reg_param in REG_PARAMS}
            release(scaled_train)
            validation = (
                read_rows(args.validation_features)
                .select(
                    "query_track_id",
                    "candidate_track_id",
                    "validation_groups",
                    *RAW_BASE_FEATURES,
                )
            )
            scaled_validation = scaler.transform(
                assembler.transform(materialize(validation))
            ).select(
                "query_track_id",
                "candidate_track_id",
                "validation_groups",
                "cos_audio",
                "cos_graph",
                "has_graph",
                "bfs_score",
                "has_bfs",
                vector_to_array("features").alias("feature_array"),
            ).persist(StorageLevel.MEMORY_AND_DISK)
            cached.append(scaled_validation)
            invalid_groups = scaled_validation.where(
                F.col("validation_groups").isNull()
                | (F.size("validation_groups") == 0)
                | F.exists(
                    "validation_groups",
                    lambda group: (
                        group["query_group"].isNull()
                        | ~group["query_group"].isin(*QUERY_GROUPS)
                        | group["label"].isNull()
                        | ~group["label"].isin(0, 1)
                        | group["eligible_positive_count"].isNull()
                        | (group["eligible_positive_count"] <= 0)
                    ),
                )
            ).limit(1).count()
            if invalid_groups:
                raise ValueError("Set-B validation contains an invalid group or denominator")
            validation_scores = _collect_validation_scores(scaled_validation, models)
            selected_reg, selection = _select_reg_configuration(validation_scores)
            model = models[selected_reg]
            release(scaled_validation)
            print(
                "set_b_ranker_diagnostics "
                + json.dumps(selection["set_b_diagnostics"], sort_keys=True),
                flush=True,
            )
        else:
            selected_reg = float(args.fixed_reg_param)
            model = fit(selected_reg)
            selection = {
                "selected_reg_param": selected_reg,
                "selection_source": "frozen_from_set_b",
                "warm_start": "tuning_ranker_coefficients",
            }

        fitted_coefficients = tuple(float(value) for value in model.coefficients)
        if args.stage == "final_retrain":
            coefficients = expand_solver_coefficients(
                solver_features,
                fitted_coefficients,
            )
        else:
            coefficients = fitted_coefficients
        constant_indexes = {
            index
            for index, name in enumerate(FEATURE_ORDER)
            if name in constant_features
        }
        if any(abs(coefficients[index]) > 1e-12 for index in constant_indexes):
            raise ValueError("LR assigned weight to a zero-variance feature")
        selection = {
            **selection,
            "training_variant": args.training_variant,
            "constant_features": list(constant_features),
            "solver_feature_count": len(solver_features),
            "max_block_size_mb": float(args.max_block_size_mb),
            "rdd_compression_codec": RDD_COMPRESSION_CODEC,
            "class_weight": "none",
        }

        parent_paths = parse_parents(args.parent)
        parent_paths.setdefault("train_features", args.train_features)
        parent_paths.setdefault("train_features_manifest", args.train_features_manifest)
        if args.stage == "final_retrain":
            assert frozen_coefficients_path is not None
            parent_paths.setdefault("frozen_scaler", args.frozen_scaler)
            parent_paths.setdefault("frozen_tuning_manifest", args.frozen_tuning_manifest)
            parent_paths.setdefault("frozen_tuning_coefficients", frozen_coefficients_path)
        if args.training_variant == "no_hard_neg":
            parent_paths.setdefault("training_pairs_manifest", pair_manifest_path)

        write_ranker_artifacts(
            args.output,
            fill_values=fill_values,
            means=means,
            stds=stds,
            coefficients=coefficients,
            intercept=float(model.intercept),
            reg_param=selected_reg,
            stage=args.stage,
            converged=True,
            iterations=int(model.summary.totalIterations),
            selection=selection,
            parent_paths=parent_paths,
            scope=args.scope,
            constant_features=constant_features,
        )
        print(
            "ranker_training_ready "
            f"stage={args.stage.removeprefix('final_')} "
            f"reg_param={selected_reg} output={args.output}",
        )
    finally:
        for frame in reversed(cached):
            try:
                frame.unpersist(blocking=False)
            except Exception:
                pass
        spark.stop()
        spark_local_temporary.cleanup()


if __name__ == "__main__":
    main()
