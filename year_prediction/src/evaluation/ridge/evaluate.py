from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

MODULE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = MODULE_DIR.parent
SOURCE_DIR = EVALUATION_DIR.parent
TRAINING_DIR = SOURCE_DIR / "training"
RIDGE_DIR = TRAINING_DIR / "ridge"
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(RIDGE_DIR))

from data import EXPECTED_COLUMNS, FEATURES, SPLIT, read_training_manifest, spark_path  # noqa: E402
from distributed import prediction_row  # noqa: E402
from metrics import (  # noqa: E402
    ABSOLUTE_ERROR_COLUMN,
    add_absolute_error,
    aggregate_decade_metrics,
    aggregate_quality_metrics,
)
from model_io import (  # noqa: E402
    prepare_output_directory,
    read_json,
    sha256_file,
    write_json,
)
from target import MAX_YEAR, MIN_YEAR, YEAR_SPAN  # noqa: E402


TEST = "test"
BASE_PREDICTION_SCHEMA = StructType(
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
PREDICTION_SCHEMA = StructType(
    [
        *BASE_PREDICTION_SCHEMA.fields,
        StructField(ABSOLUTE_ERROR_COLUMN, DoubleType(), False),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Ridge model on the test split.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("parquets/year_prediction/results/experiment_a/ridge"),
    )
    parser.add_argument("--prediction-partitions", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_finite_numbers(values: list[float], message: str) -> None:
    require(all(math.isfinite(float(value)) for value in values), message)


def ship_worker_modules(spark: SparkSession) -> None:
    for path in (
        TRAINING_DIR / "target.py",
        EVALUATION_DIR / "metrics.py",
        RIDGE_DIR / "objectives.py",
        RIDGE_DIR / "distributed.py",
    ):
        spark.sparkContext.addPyFile(str(path))


def load_model(model_directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    model_path = model_directory.resolve() / "model.json"
    model = read_json(model_path)
    require(model.get("format_version") == 1, "Unexpected Ridge model format")
    require(model.get("model_type") == "linear_ridge", "Unexpected model type")
    weights = [float(value) for value in model.get("weights", [])]
    dimension = int(model.get("feature_dimension", 0))
    require(dimension > 0 and len(weights) == dimension, "Invalid Ridge weight dimension")
    require_finite_numbers([*weights, float(model["intercept"])], "Non-finite model parameters")
    feature_source = model.get("feature_source", {})
    manifest_path = Path(feature_source["manifest"])
    require(
        sha256_file(manifest_path) == feature_source.get("manifest_sha256"),
        "T90 training manifest checksum differs",
    )
    manifest = read_training_manifest(manifest_path)
    require(model.get("target") == manifest["target"], "Model target contract differs")
    require(
        feature_source.get("contract_version") == manifest["contract_version"],
        "T90 contract version differs",
    )
    require(
        feature_source.get("predictor_order_sha256")
        == manifest["source"]["predictor_order_sha256"],
        "T90 predictor order differs",
    )
    require(
        dimension == int(manifest["preprocessing"]["dimension"]),
        "Model and T90 dimensions differ",
    )
    test_counts = manifest.get("counts", {}).get("splits", {}).get(TEST, {})
    require(int(test_counts.get("tracks", 0)) > 0, "T90 manifest has no test tracks")
    require(int(test_counts.get("artists", 0)) > 0, "T90 manifest has no test artists")
    return model, manifest


def load_test_frame(
    spark: SparkSession,
    model: dict[str, Any],
    manifest: dict[str, Any],
) -> DataFrame:
    frame = spark.read.parquet(spark_path(model["feature_source"]["input"]))
    require(tuple(frame.columns) == EXPECTED_COLUMNS, "Unexpected T90 vector columns")
    test = frame.where(F.col(SPLIT) == TEST)
    invalid_feature = F.exists(
        F.col(FEATURES),
        lambda value: value.isNull()
        | F.isnan(value)
        | (F.abs(value) == F.lit(float("inf"))),
    )
    expected_normalized_year = (
        F.col("year").cast("double") - F.lit(float(MIN_YEAR))
    ) / F.lit(float(YEAR_SPAN))
    summary = test.agg(
        F.count("*").alias("count"),
        F.countDistinct("track_id").alias("distinct_tracks"),
        F.countDistinct("artist_id").alias("distinct_artists"),
        F.min(F.size(FEATURES)).alias("minimum_dimension"),
        F.max(F.size(FEATURES)).alias("maximum_dimension"),
        F.sum(
            F.when(
                F.col(FEATURES).isNull()
                | invalid_feature
                | F.col("normalized_year").isNull()
                | F.isnan("normalized_year")
                | (F.abs(F.col("normalized_year")) == F.lit(float("inf")))
                | (F.abs(F.col("normalized_year") - expected_normalized_year) > 1.0e-12),
                1,
            ).otherwise(0)
        ).alias("invalid_rows"),
    ).first()
    require(summary is not None, "Test summary is missing")
    expected = manifest["counts"]["splits"][TEST]
    require(int(summary["count"]) == int(expected["tracks"]), "Test row count differs")
    require(
        int(summary["distinct_tracks"]) == int(expected["tracks"]),
        "Test track IDs are duplicated",
    )
    require(
        int(summary["distinct_artists"]) == int(expected["artists"]),
        "Test artist count differs",
    )
    dimension = int(model["feature_dimension"])
    require(
        int(summary["minimum_dimension"]) == dimension
        and int(summary["maximum_dimension"]) == dimension,
        "Test feature dimensions differ from the model",
    )
    require(int(summary["invalid_rows"]) == 0, "Test data contains invalid rows")
    return test

