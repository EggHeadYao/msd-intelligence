from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))

from model_io import read_json, sha256_file  # noqa: E402
from target import TARGET_COLUMN, target_contract  # noqa: E402


TRACK_ID = "track_id"
ARTIST_ID = "artist_id"
YEAR = "year"
SPLIT = "split"
FEATURES = "features"
SPLITS = ("train", "validation", "test")
OUTPUT_COLUMNS = (TRACK_ID, ARTIST_ID, YEAR, TARGET_COLUMN, FEATURES, SPLIT)
OUTPUT_CONTRACT_VERSION = "year_prediction_t90_training_v1"
VALUE_TOLERANCE = 1.0e-10
STANDARDIZATION_TOLERANCE = 1.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model-ready T90 Ridge vectors.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("parquets/year_prediction/training/t90"),
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def spark_path(path: str | Path) -> str:
    text = str(path)
    return text if "://" in text else Path(text).resolve().as_uri()


def order_sha256(columns: list[str] | tuple[str, ...]) -> str:
    payload = json.dumps(list(columns), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def schema_payload(frame: DataFrame) -> list[dict[str, str]]:
    return [
        {"name": field.name, "type": field.dataType.simpleString()}
        for field in frame.schema.fields
    ]


def finite_array(column: Column) -> Column:
    return F.forall(
        column,
        lambda value: value.isNotNull()
        & ~F.isnan(value)
        & (F.abs(value) != F.lit(float("inf"))),
    )


def validate_manifest(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    require(
        manifest.get("contract_version") == OUTPUT_CONTRACT_VERSION,
        "Unexpected T90 training contract version",
    )
    require(manifest.get("format_version") == 1, "Unexpected T90 manifest format")
    expected_target = target_contract()
    require(manifest.get("target") == expected_target, "Target contract differs")
    preprocessing = manifest.get("preprocessing", {})
    require(preprocessing.get("fit_split") == "train", "Preprocessing was not fit on train")
    require(preprocessing.get("imputation") == "train_mean", "Imputation contract differs")
    require(
        preprocessing.get("scaling") == "train_sample_standard_deviation_after_imputation",
        "Scaling contract differs",
    )
    dimension = int(preprocessing.get("dimension", 0))
    statistics = preprocessing.get("features", [])
    require(dimension == 90 and len(statistics) == dimension, "T90 dimension differs")
    names = [str(item.get("name")) for item in statistics]
    require(len(set(names)) == dimension, "T90 statistic names are duplicated")
    require(
        manifest.get("source", {}).get("predictor_order_sha256") == order_sha256(names),
        "T90 statistic order differs",
    )
    for item in statistics:
        require(math.isfinite(float(item["mean"])), f"Non-finite mean for {item['name']}")
        require(
            math.isfinite(float(item["standard_deviation"]))
            and float(item["standard_deviation"]) > 0.0,
            f"Invalid standard deviation for {item['name']}",
        )
        require(int(item["finite_train_count"]) > 0, f"Empty train feature {item['name']}")
        require(int(item["imputed_train_count"]) >= 0, f"Invalid missing count for {item['name']}")
    output = manifest.get("output", {})
    require(tuple(output.get("columns", ())) == OUTPUT_COLUMNS, "Output columns differ")
    require(output.get("partition_column") == SPLIT, "Output partition contract differs")
    return statistics, dimension

