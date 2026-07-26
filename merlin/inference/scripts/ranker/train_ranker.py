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
            functions.lit(float(fitted.intercept)),
            lambda total, value: total + value,
        )
        ranking = window.partitionBy("query_track_id").orderBy(
            functions.desc("margin"), functions.asc("candidate_track_id")
        )
        per_query = (
            frame.withColumn("margin", margin)
            .withColumn("rank", functions.row_number().over(ranking))
            .withColumn("validation_group", functions.explode("validation_groups"))
            .withColumn(
                "gain",
                functions.when(
                    (functions.col("rank") <= 20)
                    & (functions.col("validation_group.label") == 1),
                    1.0 / functions.log2(functions.col("rank") + 1.0),
                ).otherwise(0.0),
            )
            .groupBy(
                "query_track_id",
                functions.col("validation_group.query_group").alias("query_group"),
            )
            .agg(
                functions.sum("gain").alias("dcg"),
                functions.max("validation_group.eligible_positive_count").alias(
                    "positive_count"
                ),
            )
            .withColumn(
                "idcg20",
                functions.aggregate(
                    functions.sequence(
                        functions.lit(1),
                        functions.least(
                            functions.col("positive_count").cast("int"),
                            functions.lit(20),
                        ),
                    ),
                    functions.lit(0.0),
                    lambda total, rank: total
                    + 1.0 / functions.log2(rank.cast("double") + 1.0),
                ),
            )
            .withColumn("ndcg20", functions.col("dcg") / functions.col("idcg20"))
        )
        collected_by_reg[reg_param] = per_query.collect()
    return collected_by_reg


def main() -> None:
    args = parse_args()
    if args.stage == "tuning" and args.validation_features is None:
        raise ValueError("tuning requires validation features")
    if args.stage == "tuning" and args.validation_features_manifest is None:
        raise ValueError("tuning requires a validation feature manifest")
    if args.stage == "final_retrain" and args.fixed_reg_param not in REG_PARAMS:
        raise ValueError("final retrain requires one frozen regParam")
    if args.stage == "final_retrain" and (
        args.frozen_scaler is None or args.frozen_tuning_manifest is None
    ):
        raise ValueError("final retrain requires the frozen Set-A scaler and tuning manifest")
    train_feature_manifest = load_raw_feature_manifest(
        args.train_features_manifest,
        args.train_features,
        expected_scope=args.scope,
        expected_pair_kind="training",
        expected_stage=args.stage,
    )
    validation_feature_manifest = None
    frozen_preprocessing = None
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
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from pyspark.ml.functions import array_to_vector, vector_to_array
    from pyspark import StorageLevel
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    projected_gb = estimate_ranker_scratch_gb(
        training_rows=int(train_feature_manifest["row_count"]),
        validation_rows=(
            int(validation_feature_manifest["row_count"])
            if validation_feature_manifest is not None
            else 0
        ),
        feature_count=len(RANKER_V2_FEATURES),
        model_count=1,
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

        train = read_rows(args.train_features).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        cached.append(train)
        if args.stage == "tuning":
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
            inputCols=list(RANKER_V2_FEATURES),
            outputCol="unscaled_features",
            handleInvalid="error",
        )
        assembled_train = assembler.transform(materialize(train))
        if args.stage == "tuning":
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
                for name, value in zip(RANKER_V2_FEATURES, spark_stds, strict=True)
                if value == 0.0
            )
            stds = tuple(1.0 if value == 0.0 else value for value in spark_stds)
        else:
            unscaled = vector_to_array("unscaled_features")
            scaled = F.array(*(
                (unscaled[index] - F.lit(means[index])) / F.lit(stds[index])
                for index in range(len(RANKER_V2_FEATURES))
            ))
            transformed_train = assembled_train.withColumn(
                "features",
                array_to_vector(scaled),
            )
        release(train)
        scaled_train = transformed_train.select(
            F.col("label").cast("double").alias("label"), "features"
        ).persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(scaled_train)
        scaled_train.count()

        def fit(reg_param: float) -> Any:
            model = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                elasticNetParam=0.0,
                regParam=reg_param,
                maxIter=100,
                tol=1e-6,
                fitIntercept=True,
                standardization=True,
            ).fit(scaled_train)
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
            invalid_groups = validation.where(
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
            scaled_validation = scaler.transform(
                assembler.transform(materialize(validation))
            ).select(
                "query_track_id",
                "candidate_track_id",
                "validation_groups",
                vector_to_array("features").alias("feature_array"),
            ).persist(StorageLevel.MEMORY_AND_DISK)
            cached.append(scaled_validation)
            scaled_validation.count()
            collected_by_reg = _collect_validation_scores(
                scaled_validation, models, F, Window
            )
            release(scaled_validation)
            query_scores = {}
            for reg_param in REG_PARAMS:
                collected = collected_by_reg[reg_param]
                group_counts = {
                    group: sum(row["query_group"] == group for row in collected)
                    for group in QUERY_GROUPS
                }
                if any(count == 0 for count in group_counts.values()):
                    raise ValueError("Set-B validation is missing a frozen query group")
                total = len(collected)
                query_scores[reg_param] = [
                    float(row["ndcg20"])
                    * total
                    / (len(QUERY_GROUPS) * group_counts[row["query_group"]])
                    for row in sorted(
                        collected,
                        key=lambda item: (item["query_group"], item["query_track_id"]),
                    )
                ]
            selected_reg, selection = select_reg_param(query_scores)
            model = models[selected_reg]
        else:
            selected_reg = float(args.fixed_reg_param)
            model = fit(selected_reg)
            release(scaled_train)
            selection = {
                "selected_reg_param": selected_reg,
                "selection_source": "frozen_from_set_b",
            }

        coefficients = tuple(float(value) for value in model.coefficients)
        constant_indexes = {
            index
            for index, name in enumerate(RANKER_V2_FEATURES)
            if name in constant_features
        }
        if any(abs(coefficients[index]) > 1e-12 for index in constant_indexes):
            raise ValueError("LR assigned weight to a zero-variance feature")
        selection = {
            **selection,
            "constant_features": list(constant_features),
        }

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
            parent_paths=parse_parents(args.parent),
            scope=args.scope,
            constant_features=constant_features,
        )
        print(
            "ranker_training_ready "
            f"stage={args.stage} reg_param={selected_reg} output={args.output}",
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
