from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from synapse.ml.lightgbm import LightGBMRegressor

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from lightgbm_metrics import (  # noqa: E402
    add_prediction_columns,
    constant_baselines,
    decade_metrics,
    regression_metrics,
)
from lightgbm_config import file_sha256, read_config  # noqa: E402
from spark_io import write_native_model, write_parquet_parts  # noqa: E402
from spark_common import (  # noqa: E402
    assert_artist_isolation,
    load_feature_contract,
    load_feature_frame,
    prepare_output,
    split_counts,
    write_json,
)

DEFAULT_INPUT = Path("parquets/year_prediction/features/full_tabular.parquet")
DEFAULT_MANIFEST = Path("parquets/year_prediction/features/manifest.json")
DEFAULT_OUTPUT = Path("parquets/year_prediction/models/lightgbm-spark-v1")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    preliminary, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Train distributed SynapseML LightGBM")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-view", default="full_tabular")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--objective", choices=("huber", "regression"), default="huber")
    parser.add_argument("--metric", choices=("l1", "rmse"), default="l1")
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--feature-fraction", type=float, default=1.0)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--lambda-l2", type=float, default=20.0)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--bin-sample-count", type=int, default=50000)
    parser.add_argument("--decade-weight-power", type=float, default=0.0)
    parser.add_argument("--max-decade-weight", type=float, default=3.0)
    parser.add_argument("--huber-alpha", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=472)
    if preliminary.config is not None:
        allowed = {action.dest for action in parser._actions}
        parser.set_defaults(**read_config(preliminary.config, allowed))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_iterations <= 0 or args.early_stopping_rounds < 0:
        raise ValueError("iteration counts are invalid")
    if args.num_tasks < 0:
        raise ValueError("num_tasks cannot be negative")
    if args.bin_sample_count <= 0:
        raise ValueError("bin_sample_count must be positive")
    if args.decade_weight_power < 0.0 or args.max_decade_weight < 1.0:
        raise ValueError("decade weight parameters are invalid")
    if not 0.0 < args.learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    for name in ("feature_fraction", "bagging_fraction", "huber_alpha"):
        if not 0.0 < float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")


def build_estimator(args: argparse.Namespace, categorical: list[int]) -> LightGBMRegressor:
    parameters = dict(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        validationIndicatorCol="is_validation",
        objective=args.objective,
        metric=args.metric,
        alpha=args.huber_alpha,
        learningRate=args.learning_rate,
        numIterations=args.num_iterations,
        earlyStoppingRound=args.early_stopping_rounds,
        numLeaves=args.num_leaves,
        maxDepth=args.max_depth,
        minDataInLeaf=args.min_data_in_leaf,
        featureFraction=args.feature_fraction,
        baggingFraction=args.bagging_fraction,
        baggingFreq=1,
        lambdaL2=args.lambda_l2,
        maxBin=args.max_bin,
        binSampleCount=args.bin_sample_count,
        categoricalSlotIndexes=categorical,
        useMissing=True,
        zeroAsMissing=False,
        dataTransferMode="streaming",
        useSingleDatasetMode=True,
        numTasks=args.num_tasks,
        seed=args.seed,
        deterministic=True,
        verbosity=1,
    )
    if getattr(args, "decade_weight_power", 0.0) > 0.0:
        parameters["weightCol"] = "sample_weight"
    return LightGBMRegressor(**parameters)


def normalized_decade_weights(
    counts: dict[int, int], power: float, maximum: float
) -> dict[int, float]:
    largest = max(counts.values())
    relative = {
        decade: min(maximum, (largest / count) ** power)
        for decade, count in counts.items()
    }
    mean = sum(counts[decade] * weight for decade, weight in relative.items()) / sum(
        counts.values()
    )
    return {decade: weight / mean for decade, weight in relative.items()}


def fit_frame(
    train: DataFrame, validation: DataFrame, args: argparse.Namespace
) -> tuple[DataFrame, dict[int, float]]:
    if args.decade_weight_power <= 0.0:
        return train.withColumn("is_validation", F.lit(False)).unionByName(
            validation.withColumn("is_validation", F.lit(True))
        ), {}
    decade = (F.floor(F.col("year") / 10) * 10).cast("int")
    rows = train.groupBy(decade.alias("decade")).count()
    counts = {int(row["decade"]): int(row["count"]) for row in rows.collect()}
    weights = normalized_decade_weights(
        counts, args.decade_weight_power, args.max_decade_weight
    )
    mapping = F.create_map(
        *(
            item
            for decade, weight in sorted(weights.items())
            for item in (F.lit(decade), F.lit(weight))
        )
    )
    weighted_train = train.withColumn(
        "sample_weight", F.element_at(mapping, decade)
    ).withColumn("is_validation", F.lit(False))
    weighted_validation = validation.withColumn(
        "sample_weight", F.lit(1.0)
    ).withColumn("is_validation", F.lit(True))
    return weighted_train.unionByName(weighted_validation), weights


def prediction_artifact(model: Any, frame: DataFrame) -> DataFrame:
    transformed = model.transform(frame)
    return add_prediction_columns(transformed).select(
        "track_id",
        "artist_id",
        "year",
        "split",
        "raw_prediction_year",
        "clipped_prediction_year",
        "absolute_error_years",
    )


def evaluate_split(model: Any, frame: DataFrame, output: Path, split: str) -> dict[str, Any]:
    predictions = prediction_artifact(model, frame.where(F.col("split") == split)).cache()
    metrics = regression_metrics(predictions)
    metrics["by_decade"] = decade_metrics(predictions)
    write_parquet_parts(
        predictions,
        output / f"{split}_predictions.parquet",
        (
            "track_id",
            "artist_id",
            "year",
            "split",
            "raw_prediction_year",
            "clipped_prediction_year",
            "absolute_error_years",
        ),
    )
    predictions.unpersist()
    return metrics


def run(args: argparse.Namespace, spark: SparkSession) -> dict[str, Any]:
    validate_args(args)
    prepare_output(args.output, args.overwrite)
    started = time.perf_counter()
    contract = load_feature_contract(args.manifest, args.feature_view)
    frame = load_feature_frame(
        spark,
        args.input,
        contract,
        args.max_rows_per_split,
        ("train", "validation", "test") if args.evaluate_test else ("train", "validation"),
    ).repartition(
        max(1, args.num_tasks or spark.sparkContext.defaultParallelism)
    ).persist(StorageLevel.DISK_ONLY)
    assert_artist_isolation(frame)
    counts = split_counts(frame)
    train = frame.where(F.col("split") == "train")
    validation = frame.where(F.col("split") == "validation")
    training_frame, decade_weights = fit_frame(train, validation, args)
    estimator = build_estimator(args, list(contract.categorical_indexes))
    fit_started = time.perf_counter()
    model = estimator.fit(training_frame)
    fit_seconds = time.perf_counter() - fit_started
    write_native_model(args.output / "model.txt", model.getNativeModel())
    metrics: dict[str, Any] = {
        "validation": evaluate_split(model, frame, args.output, "validation")
    }
    if args.evaluate_test:
        metrics["test"] = evaluate_split(model, frame, args.output, "test")
    write_json(args.output / "metrics.json", metrics)
    write_json(args.output / "baselines.json", constant_baselines(train, validation))
    write_json(
        args.output / "feature_contract.json",
        {
            "contract_version": "year_prediction_features_v1",
            "feature_view": contract.view_name,
            "predictor_count": contract.dimension,
            "predictor_order_sha256": contract.order_sha256,
            "predictor_columns": list(contract.predictors),
            "categorical_indexes": list(contract.categorical_indexes),
        },
    )
    parameters = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    write_json(args.output / "arguments.json", parameters)
    metadata = {
        "model_type": "synapseml_lightgbm",
        "objective": args.objective,
        "metric": args.metric,
        "feature_view": contract.view_name,
        "spark_version": spark.version,
        "spark_master": spark.sparkContext.master,
        "application_id": spark.sparkContext.applicationId,
        "split_counts": counts,
        "fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - started,
        "test_read": args.evaluate_test,
    }
    if decade_weights:
        metadata["decade_weights"] = decade_weights
    if args.config is not None:
        metadata["configuration_path"] = str(args.config.resolve())
        metadata["configuration_sha256"] = file_sha256(args.config)
    write_json(args.output / "run_metadata.json", metadata)
    frame.unpersist()
    return {"metrics": metrics, "metadata": metadata}


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("YearPredictionSparkLightGBM").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        result = run(args, spark)
        print(json.dumps(result, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
