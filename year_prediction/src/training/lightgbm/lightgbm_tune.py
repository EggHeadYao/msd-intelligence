from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from lightgbm_metrics import add_prediction_columns, regression_metrics  # noqa: E402
from lightgbm_train import build_estimator  # noqa: E402
from spark_common import (  # noqa: E402
    load_feature_contract,
    load_feature_frame,
    prepare_output,
    write_json,
)

STAGES = ("capacity", "regularization", "learning_rate", "huber")
DEFAULTS: dict[str, Any] = {
    "huber_alpha": 0.9,
    "learning_rate": 0.04,
    "num_leaves": 12,
    "max_depth": 6,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "lambda_l2": 5.0,
    "max_bin": 255,
    "num_iterations": 5000,
    "early_stopping_rounds": 200,
    "seed": 472,
}
GRIDS: dict[str, tuple[dict[str, Any], ...]] = {
    "capacity": (
        {"num_leaves": 12, "max_depth": 6, "min_data_in_leaf": 500},
        {"num_leaves": 24, "max_depth": 8, "min_data_in_leaf": 300},
        {"num_leaves": 31, "max_depth": 8, "min_data_in_leaf": 200},
    ),
    "regularization": (
        {"feature_fraction": 0.85, "bagging_fraction": 0.85, "lambda_l2": 5.0},
        {"feature_fraction": 1.0, "bagging_fraction": 0.85, "lambda_l2": 20.0},
        {"feature_fraction": 0.75, "bagging_fraction": 0.75, "lambda_l2": 10.0},
    ),
    "learning_rate": (
        {"learning_rate": 0.08, "num_iterations": 2500},
        {"learning_rate": 0.04, "num_iterations": 5000},
        {"learning_rate": 0.02, "num_iterations": 8000},
    ),
    "huber": (
        {"huber_alpha": 0.8},
        {"huber_alpha": 0.9},
        {"huber_alpha": 0.95},
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune distributed Spark LightGBM")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("parquets/year_prediction/features/full_tabular.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("parquets/year_prediction/features/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parquets/year_prediction/models/lightgbm-spark-tuning"),
    )
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--num-iterations-limit", type=int)
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="ascii") as handle:
        return list(json.load(handle))


def selected_parameters(results: list[dict[str, Any]]) -> dict[str, Any]:
    parameters = dict(DEFAULTS)
    if results:
        best = min(results, key=lambda item: float(item["validation_mae_years"]))
        parameters.update(best["parameters"])
    return parameters


def run(args: argparse.Namespace, spark: SparkSession) -> list[dict[str, Any]]:
    results_path = args.output / "trials.json"
    if args.resume:
        if not args.output.exists():
            raise FileNotFoundError("cannot resume a missing tuning directory")
        results = load_results(results_path)
    else:
        prepare_output(args.output, args.overwrite)
        results = []
    contract = load_feature_contract(args.manifest)
    frame = load_feature_frame(
        spark,
        args.input,
        contract,
        args.max_rows_per_split,
        ("train", "validation"),
    ).repartition(max(1, args.num_tasks or spark.sparkContext.defaultParallelism))
    training = frame.where(F.col("split") == "train").withColumn(
        "is_validation", F.lit(False)
    )
    validation = frame.where(F.col("split") == "validation").withColumn(
        "is_validation", F.lit(True)
    )
    fit_frame = training.unionByName(validation).cache()
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        base = selected_parameters(results)
        for index, override in enumerate(GRIDS[stage]):
            parameters = {**base, **override}
            if args.num_iterations_limit is not None:
                parameters["num_iterations"] = min(
                    int(parameters["num_iterations"]), args.num_iterations_limit
                )
                parameters["early_stopping_rounds"] = min(
                    int(parameters["early_stopping_rounds"]),
                    max(1, args.num_iterations_limit // 4),
                )
            key = {"stage": stage, "index": index, "parameters": parameters}
            if any(
                item["stage"] == stage and item["index"] == index for item in results
            ):
                continue
            namespace = SimpleNamespace(**parameters, num_tasks=args.num_tasks)
            started = time.perf_counter()
            model = build_estimator(namespace, list(contract.categorical_indexes)).fit(
                fit_frame
            )
            predictions = add_prediction_columns(model.transform(validation))
            metrics = regression_metrics(predictions)
            result = {
                **key,
                "validation_mae_years": metrics["mae_years"],
                "validation_rmse_years": metrics["rmse_years"],
                "fit_seconds": time.perf_counter() - started,
            }
            results.append(result)
            write_json(results_path, results)
    write_json(args.output / "best.json", min(results, key=lambda x: x["validation_mae_years"]))
    fit_frame.unpersist()
    return results


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("YearPredictionTuneSparkLightGBM").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        results = run(args, spark)
        print(json.dumps({"trials": len(results)}, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
