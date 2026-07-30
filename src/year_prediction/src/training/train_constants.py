from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

MODULE_DIR = Path(__file__).resolve().parent
RIDGE_DIR = MODULE_DIR / "ridge"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(RIDGE_DIR))

from data import (  # noqa: E402
    EXPECTED_COLUMNS,
    SPLIT,
    TRAIN,
    VALIDATION,
    read_training_manifest,
    spark_path,
)
from model_io import write_json  # noqa: E402

TEST = "test"
MODEL_IDS = {"mean": "constant-mean", "median": "constant-median"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute year-prediction constant baselines.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("parquets/year_prediction/training/t90/vectors.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("parquets/year_prediction/training/t90/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parquets/year_prediction/results/model_comparison/constants-v1/metrics.json"),
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


def compute_constant_baselines(frame: DataFrame) -> dict[str, dict]:
    available = {str(row[SPLIT]) for row in frame.select(SPLIT).distinct().collect()}
    if not {TRAIN, VALIDATION}.issubset(available):
        raise ValueError("constant baselines require train and validation splits")
    train = frame.where(F.col("split") == TRAIN)
    statistics = train.agg(
        F.avg("year").alias("mean"),
        F.percentile_approx("year", 0.5, 10000).alias("median"),
    ).first()
    result: dict[str, dict] = {}
    for name in ("mean", "median"):
        prediction = float(statistics[name])
        model = {
            "model_id": MODEL_IDS[name],
            "fit_split": TRAIN,
            "prediction_year": prediction,
        }
        for split in (VALIDATION, TEST):
            if split in available:
                model[split] = constant_metrics(
                    frame.where(F.col(SPLIT) == split), prediction
                )
        result[MODEL_IDS[name]] = model
    return result


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("YearPredictionConstantBaselines")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        manifest = read_training_manifest(args.manifest)
        frame = spark.read.parquet(spark_path(args.input))
        if tuple(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected T90 vector columns: {frame.columns}")
        expected = {
            split: int(manifest["counts"]["splits"][split]["tracks"])
            for split in (TRAIN, VALIDATION, TEST)
        }
        actual = {
            str(row[SPLIT]): int(row["count"])
            for row in frame.groupBy(SPLIT).count().collect()
        }
        if actual != expected:
            raise ValueError("T90 split counts differ from the manifest")
        result = compute_constant_baselines(frame)
        write_json(args.output.resolve(), result)
        print(
            "year_constant_baselines "
            f"mean_test_mae={result['constant-mean']['test']['mae_years']:.6f}, "
            f"median_test_mae={result['constant-median']['test']['mae_years']:.6f}, "
            f"output={args.output.resolve()}"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
