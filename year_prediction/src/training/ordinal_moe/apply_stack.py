from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
sys.path[:0] = [str(TRAINING_DIR), str(MODULE_DIR)]

from spark_common import parquet_inputs, prepare_output, write_json  # noqa: E402
from spark_io import write_parquet_parts  # noqa: E402
from stack_predictions import prediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a frozen Spark prediction stack")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ordinal", type=Path, required=True)
    parser.add_argument("--lightgbm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, spark: SparkSession) -> dict:
    prepare_output(args.output, args.overwrite)
    with args.model.open("r", encoding="ascii") as handle:
        model = json.load(handle)
    if model.get("fit_split") != "artist_isolated_oof":
        raise ValueError("stack weights were not fitted on artist-isolated OOF data")
    ordinal = prediction(args.ordinal, "blend_year", "ordinal", spark)
    lightgbm = prediction(args.lightgbm, "clipped_prediction_year", "lightgbm", spark)
    joined = ordinal.join(lightgbm, ("track_id", "artist_id", "year"), "inner")
    result = joined.withColumn(
        "stacked_prediction_year",
        F.lit(float(model["ordinal_weight"])) * F.col("ordinal")
        + F.lit(float(model["lightgbm_weight"])) * F.col("lightgbm"),
    ).withColumn(
        "absolute_error_years", F.abs(F.col("stacked_prediction_year") - F.col("year"))
    ).cache()
    written = write_parquet_parts(result, args.output / "predictions.parquet")
    mae = float(result.agg(F.avg("absolute_error_years")).first()[0])
    metrics = {"rows": written, "mae_years": mae,
               "ordinal_weight": model["ordinal_weight"],
               "lightgbm_weight": model["lightgbm_weight"]}
    write_json(args.output / "metrics.json", metrics)
    result.unpersist()
    return metrics


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("YearPredictionApplyStack").getOrCreate()
    try:
        print(json.dumps(run(args, spark), sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
