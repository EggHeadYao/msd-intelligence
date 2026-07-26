from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pyspark.sql import SparkSession
from synapse.ml.lightgbm import LightGBMRegressionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Spark LightGBM artifacts")
    parser.add_argument("--model-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def finite_metrics(value) -> bool:
    if isinstance(value, dict):
        return all(finite_metrics(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_metrics(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    return math.isfinite(float(value))


def validate(model_root: Path, spark: SparkSession) -> dict:
    required = (
        "model.txt", "feature_contract.json", "arguments.json", "metrics.json",
        "baselines.json", "run_metadata.json", "validation_predictions.parquet",
    )
    missing = [name for name in required if not (model_root / name).exists()]
    if missing:
        raise ValueError(f"missing LightGBM artifacts: {missing}")
    contract = read_json(model_root / "feature_contract.json")
    predictor_count = contract.get("predictor_count")
    predictors = contract.get("predictor_columns", [])
    if not isinstance(predictor_count, int) or predictor_count <= 0:
        raise ValueError("unexpected LightGBM feature dimension")
    if len(predictors) != predictor_count or len(set(predictors)) != predictor_count:
        raise ValueError("LightGBM predictor list is invalid")
    metrics = read_json(model_root / "metrics.json")
    if not finite_metrics(metrics):
        raise ValueError("LightGBM metrics contain non-finite values")
    metadata = read_json(model_root / "run_metadata.json")
    if metadata.get("model_type") not in {
        "synapseml_lightgbm_huber",
        "synapseml_lightgbm",
    }:
        raise ValueError("unexpected LightGBM model type")
    model_text = (model_root / "model.txt").read_text(encoding="ascii")
    model = LightGBMRegressionModel.loadNativeModelFromString(model_text)
    if len(model.getNativeModel()) < 100:
        raise ValueError("native LightGBM model is empty")
    return {"valid": True, "spark_version": spark.version,
            "validation_rows": metrics["validation"]["count"]}


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("ValidateSparkLightGBM").getOrCreate()
    try:
        print(json.dumps(validate(args.model_root, spark), sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
