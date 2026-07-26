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


def validate_view(
    frame: DataFrame,
    predictors: list[str],
    expected_hash: str,
) -> dict[str, Any]:
    require(frame.columns == [*AUDIT_COLUMNS, *predictors], "column order differs")
    require(len(set(predictors)) == len(predictors), "predictors are duplicated")
    require(order_sha256(predictors) == expected_hash, "predictor hash differs")
    require(not set(AUDIT_COLUMNS) & set(predictors), "audit column is a predictor")
    summary = frame.agg(
        F.count("*").alias("rows"),
        F.countDistinct("track_id").alias("tracks"),
        F.sum(
            F.when(
                F.col("track_id").isNull()
                | (F.col("track_id") == "")
                | F.col("artist_id").isNull()
                | (F.col("artist_id") == ""),
                1,
            ).otherwise(0)
        ).alias("invalid_ids"),
    ).first()
    expected_rows = sum(EXPECTED_SPLITS.values())
    require(int(summary["rows"]) == expected_rows, "row count differs")
    require(int(summary["tracks"]) == expected_rows, "track IDs are duplicated")
    require(int(summary["invalid_ids"]) == 0, "invalid IDs found")
    split_rows = {
        str(row["split"]): int(row["count"])
        for row in frame.groupBy("split").count().collect()
    }
    require(split_rows == EXPECTED_SPLITS, "split counts differ")
    artists = {
        split: frame.where(F.col("split") == split).select("artist_id").distinct()
        for split in EXPECTED_SPLITS
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        require(
            artists[left].intersect(artists[right]).limit(1).count() == 0,
            f"artist overlap between {left} and {right}",
        )
    return {"rows": expected_rows, "splits": split_rows}

