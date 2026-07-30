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


def coverage(frame: DataFrame) -> list[dict[str, Any]]:
    rows = frame.groupBy("split").agg(
        F.avg(F.when(F.col("term_count") > 0, 1.0).otherwise(0.0)).alias("terms"),
        F.avg(F.when(F.col("mbtag_count") > 0, 1.0).otherwise(0.0)).alias("mbtags"),
        F.avg(F.when(F.col("tag_era_count") > 0, 1.0).otherwise(0.0)).alias("era_tags"),
        F.avg(
            F.when(F.col("similar_train_artist_count") > 0, 1.0).otherwise(0.0)
        ).alias("train_neighbors"),
        F.avg(F.when(F.col("artist_location_missing") == 0, 1.0).otherwise(0.0)).alias(
            "location"
        ),
    ).orderBy("split").collect()
    return [
        {
            key: str(row[key]) if key == "split" else float(row[key])
            for key in row.asDict()
        }
        for row in rows
    ]


def run(args: argparse.Namespace, spark: SparkSession) -> dict[str, Any]:
    with (args.input / "manifest.json").open(encoding="ascii") as handle:
        manifest = json.load(handle)
    require(
        manifest.get("contract_version") == "year_prediction_features_v1",
        "contract version differs",
    )
    reports: dict[str, Any] = {}
    frames: dict[str, DataFrame] = {}
    for name in ("metadata_only", "audio_metadata"):
        view = manifest["views"][name]
        frame = spark.read.parquet(spark_path(args.input / view["path"]))
        frames[name] = frame
        reports[name] = validate_view(
            frame,
            list(view["predictor_columns"]),
            str(view["predictor_order_sha256"]),
        )
    reports["coverage"] = coverage(frames["metadata_only"])
    return reports


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("ValidateYearMetadataFeatures").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        print(json.dumps(run(args, spark), indent=2, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
