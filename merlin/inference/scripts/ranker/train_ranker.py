"""Train the frozen Spark LR-L2 Ranker and publish Python inference artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from ...artifacts.integrity import sha256_path
from ...artifacts.paths import InferenceArtifactPaths
from ...ranking.model import write_ranker_artifacts
from ...ranking.features import (
    FEATURE_ORDER,
    FEATURE_SCHEMA,
    FILL_FEATURES,
    RAW_BASE_FEATURES,
    load_raw_feature_manifest,
)
from ...ranking.selection import (
    REG_PARAMS,
    select_grouped_reg_param,
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
    columns: Mapping[object, tuple[float, ...]]

    def column(self, key: object) -> tuple[float, ...]:
        values = self.columns[key]
        if len(values) != len(self.query_ids):
            raise ValueError("Set-B score column length mismatch")
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-features-manifest", type=Path, required=True)
    parser.add_argument("--training-pairs-manifest", type=Path)
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


def _idcg20(positive_count: int) -> float:
    return sum(
        1.0 / math.log2(rank + 1.0)
        for rank in range(1, min(int(positive_count), 20) + 1)
    )


def load_frozen_preprocessing(
    scaler_path: Path,
    tuning_manifest_path: Path,
    fixed_reg_param: float,
) -> tuple[dict[str, float], tuple[float, ...], tuple[float, ...], tuple[str, ...]]:
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
    return fill_values, means, stds, constant_features


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
    coefficient_rows = tuple(
        tuple(float(value) for value in models[reg_param].coefficients)
        for reg_param in REG_PARAMS
    )
    intercepts = tuple(float(models[reg_param].intercept) for reg_param in REG_PARAMS)
    model_names = tuple(f"reg:{reg_param:.17g}" for reg_param in REG_PARAMS)
    reg_by_scorer = dict(zip(model_names, REG_PARAMS, strict=True))
    scorer_names = model_names + ("c1_only", "c2_only", "bfs")

    def score_query(query):
        import numpy as np
        import pandas as pd

        query_id = str(query["query_track_id"].iloc[0])
        candidate_ids = query["candidate_track_id"].astype(str).to_numpy()
        feature_matrix = np.vstack(query["feature_array"].to_numpy()).astype(
            np.float64, copy=False
        )
        lr_score_matrix = (
            feature_matrix @ np.asarray(coefficient_rows, dtype=np.float64).T
            + np.asarray(intercepts, dtype=np.float64)
        )
        audio_scores = query["cos_audio"].to_numpy(dtype=np.float64)
        scores = {
            **{
                name: lr_score_matrix[:, model_index]
                for model_index, name in enumerate(model_names)
            },
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

        metrics = {name: {} for name in scorer_names}
        discounts = 1.0 / np.log2(np.arange(2, 22, dtype=np.float64))
        for scorer, margins in scores.items():
            top = top_indices(margins)
            for group, positive_count in positive_counts.items():
                dcg = float(np.sum(discounts[: len(top)] * labels[group][top]))
                ndcg20 = dcg / _idcg20(positive_count)
                metrics[scorer][group] = ndcg20
        rows = [
            {
                "query_track_id": query_id,
                "query_group": group,
                "score_values": [metrics[name][group] for name in scorer_names],
            }
            for group in positive_counts
        ]
        return pd.DataFrame.from_records(rows)

    rows = (
        frame.groupBy("query_track_id")
        .applyInPandas(
            score_query,
            "query_track_id string, query_group string, score_values array<double>",
        )
        .collect()
    )
    keys = tuple(reg_by_scorer.get(name, name) for name in scorer_names)
    collected = {key: [] for key in keys}
    query_ids = []
    query_groups = []
    for row in rows:
        values = row["score_values"]
        if len(values) != len(keys):
            raise ValueError("Set-B wide score vector length mismatch")
        query_ids.append(str(row["query_track_id"]))
        query_groups.append(str(row["query_group"]))
        for key, value in zip(keys, values, strict=True):
            collected[key].append(float(value))
    return ValidationScoreTable(
        query_ids=tuple(query_ids),
        query_groups=tuple(query_groups),
        columns={key: tuple(values) for key, values in collected.items()},
    )


def _validation_summary(
    table: ValidationScoreTable,
    key: object,
) -> dict[str, object]:
    values = table.column(key)
    grouped = {
        group: [
            value
            for value, query_group in zip(values, table.query_groups, strict=True)
            if query_group == group
        ]
        for group in QUERY_GROUPS
    }
    if any(not values for values in grouped.values()):
        raise ValueError("Set-B validation is missing a frozen query group")
    by_group = {
        group: sum(values) / len(values) for group, values in grouped.items()
    }
    return {
        "by_group": by_group,
        "three_strata_macro": sum(by_group.values()) / len(QUERY_GROUPS),
    }


def _grouped_query_scores(
    table: ValidationScoreTable,
    key: object,
) -> dict[str, dict[str, float]]:
    grouped = {group: {} for group in QUERY_GROUPS}
    for query_id, group, value in zip(
        table.query_ids,
        table.query_groups,
        table.column(key),
        strict=True,
    ):
        if group not in grouped:
            raise ValueError(f"unknown Set-B query group: {group}")
        if query_id in grouped[group]:
            raise ValueError("duplicate Set-B query-group score")
        grouped[group][query_id] = value
    if any(not scores for scores in grouped.values()):
        raise ValueError("Set-B validation is missing a frozen query group")
    return grouped


def _select_reg_configuration(
    validation_scores: ValidationScoreTable,
) -> tuple[float, dict[str, object]]:
    """Choose the sole tunable Ranker parameter from Set-B query metrics."""
    summaries = {
        reg_param: _validation_summary(validation_scores, reg_param)
        for reg_param in REG_PARAMS
    }
    c1_summary = _validation_summary(validation_scores, "c1_only")
    c2_summary = _validation_summary(validation_scores, "c2_only")
    bfs_summary = _validation_summary(validation_scores, "bfs")
    query_scores = {
        reg_param: _grouped_query_scores(validation_scores, reg_param)
        for reg_param in REG_PARAMS
    }
    selected_reg, report = select_grouped_reg_param(query_scores)
    report["set_b_diagnostics"] = {
        "full": summaries[selected_reg],
        "c1_only": c1_summary,
        "c2_only": c2_summary,
        "bfs": bfs_summary,
    }
    report["configuration_metrics"] = {
        f"reg={reg_param:g}": summaries[reg_param] for reg_param in REG_PARAMS
    }
    return selected_reg, report


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
    if args.training_variant == "no_hard_neg":
        paths = InferenceArtifactPaths()
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
        training_rows=int(train_feature_manifest["row_count"]),
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

        train = read_rows(args.train_features).select("label", *RAW_BASE_FEATURES)
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
            fill_values, means, stds, constant_features = frozen_preprocessing

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
