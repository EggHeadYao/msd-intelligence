from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

TRAINING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAINING_DIR))

from spark_common import parquet_inputs, prepare_output, write_json  # noqa: E402
from spark_io import write_parquet_parts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a two-model Spark OOF stack")
    parser.add_argument("--ordinal", type=Path, required=True)
    parser.add_argument("--lightgbm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prediction(path: Path, source: str, alias: str, spark: SparkSession):
    frame = spark.read.parquet(*parquet_inputs(path))
    required = {"track_id", "artist_id", "year", source}
    if not required.issubset(frame.columns):
        raise ValueError(f"prediction input lacks columns: {sorted(required)}")
    return frame.select("track_id", "artist_id", "year", F.col(source).alias(alias))


def run(args: argparse.Namespace, spark: SparkSession) -> dict:
    prepare_output(args.output, args.overwrite)
    ordinal = prediction(args.ordinal, "blend_year", "ordinal", spark)
    lightgbm = prediction(args.lightgbm, "clipped_prediction_year", "lightgbm", spark)
    joined = ordinal.join(lightgbm, ("track_id", "artist_id", "year"), "inner").cache()
    summary = joined.agg(
        F.count("*").alias("rows"), F.countDistinct("track_id").alias("tracks"),
        F.sum((F.col("ordinal") - F.col("lightgbm"))
              * (F.col("year") - F.col("lightgbm"))).alias("numerator"),
        F.sum(F.pow(F.col("ordinal") - F.col("lightgbm"), 2)).alias("denominator"),
    ).first()
    if summary is None or summary["rows"] != summary["tracks"] or summary["rows"] <= 0:
        raise ValueError("OOF predictions are empty or duplicated")
    denominator = float(summary["denominator"])
    ordinal_weight = 0.5 if denominator <= 1.0e-12 else max(
        0.0, min(1.0, float(summary["numerator"]) / denominator)
    )
    lightgbm_weight = 1.0 - ordinal_weight
    stacked = joined.withColumn(
        "stacked_prediction_year",
        F.lit(ordinal_weight) * F.col("ordinal")
        + F.lit(lightgbm_weight) * F.col("lightgbm"),
    ).withColumn(
        "absolute_error_years", F.abs(F.col("stacked_prediction_year") - F.col("year"))
    )
    write_parquet_parts(stacked, args.output / "oof_predictions.parquet")
    mae = float(stacked.agg(F.avg("absolute_error_years")).first()[0])
    model = {"fit_split": "artist_isolated_oof", "loss": "squared_error",
             "ordinal_weight": ordinal_weight, "lightgbm_weight": lightgbm_weight,
             "rows": int(summary["rows"]), "oof_mae_years": mae}
    write_json(args.output / "stack_model.json", model)
    joined.unpersist()
    return model


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("YearPredictionOOFStack").getOrCreate()
    try:
        print(json.dumps(run(args, spark), sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
