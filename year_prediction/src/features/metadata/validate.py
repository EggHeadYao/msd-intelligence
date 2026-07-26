from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

AUDIT_COLUMNS = ("track_id", "artist_id", "year", "split")
EXPECTED_SPLITS = {"train": 420_013, "validation": 46_127, "test": 49_436}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate metadata feature views")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("parquets/year_prediction/features/metadata"),
    )
    return parser.parse_args()


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def order_sha256(columns: list[str]) -> str:
    payload = json.dumps(columns, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

