from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

MODULE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = MODULE_DIR.parent / "evaluation"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(EVALUATION_DIR))

from distributed import (  # noqa: E402
    direct_full_batch_statistics,
    evaluate_linear_model,
    prediction_row,
)
from model_io import (  # noqa: E402
    prepare_output_directory,
    read_json,
    sha256_file,
    write_json,
)
from optimizer import gradient_norm, gradient_step  # noqa: E402
from training_data import (  # noqa: E402
    TRAIN,
    VALIDATION,
    load_training_data,
    read_feature_metadata,
    spark_path,
)


PREDICTION_SCHEMA = StructType(
    [
        StructField("track_id", StringType(), False),
        StructField("artist_id", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("normalized_year", DoubleType(), False),
        StructField("normalized_prediction", DoubleType(), False),
        StructField("raw_prediction_year", DoubleType(), False),
        StructField("clipped_prediction_year", DoubleType(), False),
    ]
)


def default_config_path() -> Path:
    return MODULE_DIR.parents[1] / "config" / "experiment_b" / "d0_direct.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configurable Spark SGD model.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--model-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_number(config: dict[str, Any], name: str, minimum: float, strict: bool) -> float:
    value = float(config[name])
    valid = value > minimum if strict else value >= minimum
    if not valid or not math.isfinite(value):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "model_id",
        "input",
        "feature_metadata",
        "output_root",
        "objective",
        "initialization",
        "max_iterations",
        "learning_rate",
        "l2",
        "gradient_tolerance",
        "validation_interval",
        "shuffle_partitions",
        "prediction_partitions",
        "execution",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing training configuration fields: {missing}")
    if config["objective"] != "ridge_squared":
        raise ValueError("train_sgd currently supports objective=ridge_squared")
    if config["initialization"] not in {"zeros", "zero_weights_train_mean_intercept"}:
        raise ValueError("Unsupported initialization")
    for name in ("max_iterations", "validation_interval", "shuffle_partitions", "prediction_partitions"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    require_number(config, "learning_rate", 0.0, strict=True)
    require_number(config, "l2", 0.0, strict=False)
    require_number(config, "gradient_tolerance", 0.0, strict=False)
    execution = config["execution"]
    expected = {
        "aggregation": "direct_reduce",
        "batch_fraction": 1.0,
        "broadcast_weights": False,
        "persist_training_data": False,
    }
    if execution != expected:
        raise ValueError(f"Unsupported execution configuration: expected={expected}")


def load_config(
    path: Path,
    model_id: str | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    config = read_json(path.resolve())
    if model_id is not None:
        config["model_id"] = model_id
    if output_root is not None:
        config["output_root"] = str(output_root)
    validate_config(config)
    return config


def ship_worker_modules(spark: SparkSession) -> None:
    for path in (
        EVALUATION_DIR / "metrics.py",
        MODULE_DIR / "objectives.py",
        MODULE_DIR / "distributed.py",
        MODULE_DIR / "training_data.py",
    ):
        spark.sparkContext.addPyFile(str(path))


def train(
    config: dict[str, Any],
    spark: SparkSession,
    config_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    validate_config(config)
    total_started = time.perf_counter()
    output = Path(config["output_root"]) / str(config["model_id"])
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output.resolve()}")
    metadata_path = Path(config["feature_metadata"]).resolve()
    feature_metadata = read_feature_metadata(metadata_path)
    data_started = time.perf_counter()
    data = load_training_data(spark, config["input"], feature_metadata)
    data_validation_seconds = time.perf_counter() - data_started
    ship_worker_modules(spark)
    training_points = data.points(TRAIN)
    validation_points = data.points(VALIDATION)
    weights = [0.0] * data.dimension
    intercept = (
        data.label_means[TRAIN]
        if config["initialization"] == "zero_weights_train_mean_intercept"
        else 0.0
    )
    history: list[dict[str, Any]] = []
    optimizer_started = time.perf_counter()
    stop_reason = "max_iterations"
    for iteration in range(1, int(config["max_iterations"]) + 1):
        iteration_started = time.perf_counter()
        gradient_started = time.perf_counter()
        statistics = direct_full_batch_statistics(
            training_points,
            weights,
            intercept,
            float(config["l2"]),
        )
        gradient_seconds = time.perf_counter() - gradient_started
        norm = gradient_norm(statistics.gradient, statistics.intercept_gradient)
        converged = norm <= float(config["gradient_tolerance"])
        update_started = time.perf_counter()
        if not converged:
            weights, intercept = gradient_step(
                weights,
                intercept,
                statistics.gradient,
                statistics.intercept_gradient,
                float(config["learning_rate"]),
            )
        update_seconds = time.perf_counter() - update_started
        validation = None
        validation_seconds = 0.0
        if iteration % int(config["validation_interval"]) == 0 or converged:
            validation_started = time.perf_counter()
            validation = evaluate_linear_model(validation_points, weights, intercept).as_dict()
            validation_seconds = time.perf_counter() - validation_started
        record = {
            "iteration": iteration,
            "training_objective_before_update": statistics.objective,
            "gradient_norm_before_update": norm,
            "training_count": statistics.count,
            "learning_rate": float(config["learning_rate"]),
            "updated": not converged,
            "validation_after_update": validation,
            "gradient_seconds": gradient_seconds,
            "update_seconds": update_seconds,
            "validation_seconds": validation_seconds,
            "iteration_seconds": time.perf_counter() - iteration_started,
        }
        history.append(record)
        validation_text = "not_evaluated" if validation is None else f"{validation['mae_years']:.6f}"
        print(
            "ridge_iteration "
            f"iteration={iteration}, objective={statistics.objective:.12f}, "
            f"gradient_norm={norm:.12f}, validation_mae={validation_text}, "
            f"seconds={record['iteration_seconds']:.6f}"
        )
        if converged:
            stop_reason = "gradient_tolerance"
            break
    optimizer_seconds = time.perf_counter() - optimizer_started
    final_evaluation_started = time.perf_counter()
    final_statistics = direct_full_batch_statistics(
        training_points,
        weights,
        intercept,
        float(config["l2"]),
    )
    final_training_metrics = evaluate_linear_model(training_points, weights, intercept)
    final_validation_metrics = evaluate_linear_model(validation_points, weights, intercept)
    final_evaluation_seconds = time.perf_counter() - final_evaluation_started
    output = prepare_output_directory(output, overwrite=overwrite)
    prediction_started = time.perf_counter()
    prediction_rdd = data.prediction_rows(VALIDATION).map(
        lambda row: prediction_row(row, weights, intercept)
    )
    predictions = spark.createDataFrame(prediction_rdd, PREDICTION_SCHEMA)
    predictions.repartition(int(config["prediction_partitions"])).write.mode("error").parquet(
        spark_path(output / "validation_predictions.parquet")
    )
    prediction_seconds = time.perf_counter() - prediction_started
    model = {
        "format_version": 1,
        "model_id": config["model_id"],
        "model_type": "linear_ridge",
        "objective": config["objective"],
        "feature_dimension": data.dimension,
        "weights": weights,
        "intercept": intercept,
        "l2": float(config["l2"]),
        "target": feature_metadata["target"],
        "feature_source": {
            "input": str(Path(config["input"]).resolve()),
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "feature_version": feature_metadata["feature_version"],
        },
    }
    metrics = {
        "stop_reason": stop_reason,
        "iterations_completed": len(history),
        "final_training_objective": final_statistics.objective,
        "final_gradient_norm": gradient_norm(
            final_statistics.gradient, final_statistics.intercept_gradient
        ),
        "train": final_training_metrics.as_dict(),
        "validation": final_validation_metrics.as_dict(),
    }
    run_metadata = {
        "configuration": config,
        "configuration_path": None if config_path is None else str(config_path.resolve()),
        "configuration_sha256": None if config_path is None else sha256_file(config_path.resolve()),
        "counts": data.counts,
        "spark": {
            "application_id": spark.sparkContext.applicationId,
            "master": spark.sparkContext.master,
            "version": spark.version,
            "default_parallelism": spark.sparkContext.defaultParallelism,
            "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        },
        "timing_seconds": {
            "data_validation": data_validation_seconds,
            "training_loop": optimizer_seconds,
            "gradient_reduce": sum(row["gradient_seconds"] for row in history),
            "parameter_update": sum(row["update_seconds"] for row in history),
            "validation_during_training": sum(row["validation_seconds"] for row in history),
            "final_evaluation": final_evaluation_seconds,
            "prediction_write": prediction_seconds,
            "total": time.perf_counter() - total_started,
        },
    }
    write_json(output / "model.json", model)
    write_json(output / "history.json", history)
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_metadata.json", run_metadata)
    print(
        "year_ridge_trained "
        f"model_id={config['model_id']}, iterations={len(history)}, "
        f"validation_mae={final_validation_metrics.mae_years:.6f}, output={output}"
    )
    return output


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path, args.model_id, args.output_root)
    spark = (
        SparkSession.builder.appName(f"YearPredictionTrain-{config['model_id']}")
        .config("spark.sql.shuffle.partitions", str(config["shuffle_partitions"]))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        train(config, spark, config_path=config_path, overwrite=args.overwrite)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
