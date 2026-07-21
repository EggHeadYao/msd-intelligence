from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
LIGHTGBM_DIR = TRAINING_DIR / "lightgbm"
sys.path[:0] = [str(TRAINING_DIR), str(LIGHTGBM_DIR)]

from lightgbm_metrics import add_prediction_columns, regression_metrics  # noqa: E402
from lightgbm_train import build_estimator  # noqa: E402
from spark_common import (  # noqa: E402
    assert_artist_isolation,
    load_feature_contract,
    load_feature_frame,
    parquet_inputs,
    prepare_output,
    write_json,
)
from spark_io import write_native_model, write_parquet_parts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Spark LightGBM OOF fold")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--feature-fraction", type=float, default=1.0)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--lambda-l2", type=float, default=20.0)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--huber-alpha", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=472)
    return parser.parse_args()


def run(args: argparse.Namespace, spark: SparkSession) -> dict:
    prepare_output(args.output, args.overwrite)
    contract = load_feature_contract(args.manifest)
    source = load_feature_frame(spark, args.input, contract, splits=("train",))
    folds = spark.read.parquet(*parquet_inputs(args.folds)).select("artist_id", "fold")
    frame = source.join(folds, "artist_id", "inner").withColumn(
        "split", F.when(F.col("fold") == args.fold, "validation").otherwise("train")
    ).drop("fold")
    assert_artist_isolation(frame)
    counts = {row["split"]: row["count"] for row in frame.groupBy("split").count().collect()}
    if counts.get("train", 0) <= 0 or counts.get("validation", 0) <= 0:
        raise ValueError("OOF training and held-out folds must both be non-empty")
    train = frame.where(F.col("split") == "train").withColumn(
        "is_validation", F.lit(False)
    )
    held_out = frame.where(F.col("split") == "validation").cache()
    fit_frame = train.unionByName(held_out.withColumn("is_validation", F.lit(True)))
    model = build_estimator(args, list(contract.categorical_indexes)).fit(fit_frame)
    predictions = add_prediction_columns(model.transform(held_out)).select(
        "track_id", "artist_id", "year", "raw_prediction_year",
        "clipped_prediction_year", "absolute_error_years",
    ).withColumn("fold", F.lit(args.fold)).cache()
    write_native_model(args.output / "model.txt", model.getNativeModel())
    rows = write_parquet_parts(predictions, args.output / "predictions.parquet")
    metrics = regression_metrics(predictions)
    metrics.update({"fold": args.fold, "rows": rows})
    write_json(args.output / "metrics.json", metrics)
    predictions.unpersist()
    held_out.unpersist()
    return metrics


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("YearPredictionLightGBMOOF").getOrCreate()
    try:
        print(json.dumps(run(args, spark), sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
