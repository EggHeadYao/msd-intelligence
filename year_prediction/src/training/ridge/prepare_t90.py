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

from model_io import read_json, sha256_file, write_json  # noqa: E402
from target import TARGET_COLUMN, target_contract  # noqa: E402


TRACK_ID = "track_id"
ARTIST_ID = "artist_id"
YEAR = "year"
SPLIT = "split"
FEATURES = "features"
AUDIT_COLUMNS = (TRACK_ID, ARTIST_ID, YEAR, SPLIT)
SPLITS = ("train", "validation", "test")
SOURCE_CONTRACT_VERSION = "year_prediction_features_v1"
OUTPUT_CONTRACT_VERSION = "year_prediction_t90_training_v1"
T90_DIMENSION = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare model-ready T90 Ridge vectors.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("parquets/year_prediction/features/t90.parquet"),
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("parquets/year_prediction/features/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parquets/year_prediction/training/t90"),
    )
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    parser.add_argument("--output-partitions", type=int, default=32)
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

