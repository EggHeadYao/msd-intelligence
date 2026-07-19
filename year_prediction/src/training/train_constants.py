from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from model_io import write_json  # noqa: E402
from training_data import TRAIN, VALIDATION, load_training_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute year-prediction constant baselines.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("parquets/year_prediction/features/k1/linear_vectors.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parquets/year_prediction/results/experiment_a/constants-v1/metrics.json"),
    )
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    return parser.parse_args()


def constant_metrics(validation: DataFrame, prediction: float) -> dict[str, float | int]:
    error = F.lit(float(prediction)) - F.col("year").cast("double")
    row = validation.agg(
        F.count(F.lit(1)).alias("count"),
        F.avg(F.abs(error)).alias("mae_years"),
        F.sqrt(F.avg(error * error)).alias("rmse_years"),
        F.avg(error).alias("signed_error_years"),
    ).first()
    result = {
        "count": int(row["count"]),
        "mae_years": float(row["mae_years"]),
        "rmse_years": float(row["rmse_years"]),
        "signed_error_years": float(row["signed_error_years"]),
        "prediction_year": float(prediction),
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise ValueError("constant baseline produced a non-finite metric")
    return result


def compute_constant_baselines(frame: DataFrame) -> dict[str, dict[str, float | int]]:
    train = frame.where(F.col("split") == TRAIN)
    validation = frame.where(F.col("split") == VALIDATION)
    statistics = train.agg(
        F.avg("year").alias("mean"),
        F.percentile_approx("year", 0.5, 10000).alias("median"),
    ).first()
    mean = float(statistics["mean"])
    median = float(statistics["median"])
    return {
        "mean": constant_metrics(validation, mean),
        "median": constant_metrics(validation, median),
    }


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("YearPredictionConstantBaselines")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        data = load_training_data(spark, args.input)
        result = compute_constant_baselines(data.frame)
        write_json(args.output.resolve(), result)
        print(
            "year_constant_baselines "
            f"mean_mae={result['mean']['mae_years']:.6f}, "
            f"median_mae={result['median']['mae_years']:.6f}, output={args.output.resolve()}"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
