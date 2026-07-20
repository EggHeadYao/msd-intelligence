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


def schema_payload(frame: DataFrame) -> list[dict[str, str]]:
    return [
        {"name": field.name, "type": field.dataType.simpleString()}
        for field in frame.schema.fields
    ]


def source_columns(manifest: dict[str, Any]) -> tuple[str, ...]:
    require(
        manifest.get("contract_version") == SOURCE_CONTRACT_VERSION,
        "Unexpected feature contract version",
    )
    require(tuple(manifest.get("audit_columns", ())) == AUDIT_COLUMNS, "Audit columns differ")
    view = manifest.get("views", {}).get("t90", {})
    columns = tuple(view.get("predictor_columns", ()))
    require(len(columns) == T90_DIMENSION, "T90 feature count differs")
    require(len(set(columns)) == T90_DIMENSION, "T90 feature names are duplicated")
    require(view.get("predictor_count") == T90_DIMENSION, "T90 manifest dimension differs")
    require(
        view.get("predictor_order_sha256") == order_sha256(columns),
        "T90 feature order hash differs",
    )
    return columns


def require_source_schema(frame: DataFrame, columns: tuple[str, ...]) -> None:
    require(tuple(frame.columns) == AUDIT_COLUMNS + columns, "T90 source column order differs")
    types = {field.name: field.dataType.simpleString() for field in frame.schema.fields}
    expected = {
        TRACK_ID: "string",
        ARTIST_ID: "string",
        YEAR: "int",
        SPLIT: "string",
        **{column: "double" for column in columns},
    }
    require(types == expected, "T90 source schema differs")


def require_labeled_source(frame: DataFrame, manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    labeled = frame.where(F.col(SPLIT).isNotNull())
    target = target_contract()
    values = F.array(*(F.col(column) for column in frame.columns[len(AUDIT_COLUMNS) :]))
    non_finite = F.exists(
        values,
        lambda value: value.isNotNull()
        & (F.isnan(value) | (F.abs(value) == F.lit(float("inf")))),
    )
    invalid = (
        F.col(TRACK_ID).isNull()
        | (F.col(TRACK_ID) == "")
        | F.col(ARTIST_ID).isNull()
        | (F.col(ARTIST_ID) == "")
        | F.col(YEAR).isNull()
        | ~F.col(YEAR).between(target["minimum"], target["maximum"])
        | ~F.col(SPLIT).isin(*SPLITS)
        | non_finite
    )
    rows = labeled.groupBy(SPLIT).agg(
        F.count("*").alias("tracks"),
        F.countDistinct(TRACK_ID).alias("distinct_tracks"),
        F.countDistinct(ARTIST_ID).alias("artists"),
        F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid"),
    ).collect()
    counts = {
        row[SPLIT]: {"tracks": int(row["tracks"]), "artists": int(row["artists"])}
        for row in rows
    }
    expected = {
        name: {
            "tracks": int(values["tracks"]),
            "artists": int(values["artists"]),
        }
        for name, values in manifest["counts"]["splits"].items()
    }
    require(counts == expected, "T90 source split counts differ")
    require(
        all(int(row["tracks"]) == int(row["distinct_tracks"]) for row in rows),
        "Track IDs are duplicated",
    )
    require(all(int(row["invalid"]) == 0 for row in rows), "T90 source contains invalid rows")
    require(
        labeled.select(TRACK_ID).distinct().count() == sum(item["tracks"] for item in counts.values()),
        "Track IDs overlap across splits",
    )
    overlap = (
        labeled.groupBy(ARTIST_ID)
        .agg(F.countDistinct(SPLIT).alias("split_count"))
        .where(F.col("split_count") > 1)
        .limit(1)
        .count()
    )
    require(overlap == 0, "Artists overlap across splits")
    return counts


def fit_feature_statistics(
    labeled: DataFrame,
    columns: tuple[str, ...],
) -> list[dict[str, int | float | str]]:
    train = labeled.where(F.col(SPLIT) == "train")
    mean_row = train.agg(
        F.count("*").alias("_rows"),
        *(F.count(column).alias(f"_count_{index}") for index, column in enumerate(columns)),
        *(F.avg(column).alias(f"_mean_{index}") for index, column in enumerate(columns)),
    ).first()
    train_count = int(mean_row["_rows"])
    require(train_count > 1, "Training split must contain at least two rows")
    means: list[float] = []
    finite_counts: list[int] = []
    for index, column in enumerate(columns):
        mean = mean_row[f"_mean_{index}"]
        count = int(mean_row[f"_count_{index}"])
        require(
            mean is not None and math.isfinite(float(mean)),
            f"No finite train mean for {column}",
        )
        means.append(float(mean))
        finite_counts.append(count)
    imputed = train.select(
        *(
            F.coalesce(F.col(column), F.lit(mean)).alias(column)
            for column, mean in zip(columns, means)
        )
    )
    std_row = imputed.agg(
        *(F.stddev_samp(column).alias(f"_std_{index}") for index, column in enumerate(columns))
    ).first()
    statistics: list[dict[str, int | float | str]] = []
    for index, (column, mean) in enumerate(zip(columns, means)):
        std = std_row[f"_std_{index}"]
        require(
            std is not None and math.isfinite(float(std)) and float(std) > 0.0,
            f"Training standard deviation must be positive for {column}",
        )
        statistics.append(
            {
                "name": column,
                "mean": mean,
                "standard_deviation": float(std),
                "finite_train_count": finite_counts[index],
                "imputed_train_count": train_count - finite_counts[index],
            }
        )
    return statistics


def transform(frame: DataFrame, statistics: list[dict[str, int | float | str]]) -> DataFrame:
    values = [
        (
            (F.coalesce(F.col(str(item["name"])), F.lit(float(item["mean"]))) - float(item["mean"]))
            / float(item["standard_deviation"])
        ).cast("double")
        for item in statistics
    ]
    target = target_contract()
    return frame.select(
        TRACK_ID,
        ARTIST_ID,
        YEAR,
        ((F.col(YEAR).cast("double") - target["minimum"]) / target["span"]).alias(TARGET_COLUMN),
        F.array(*values).alias(FEATURES),
        SPLIT,
    )


def build(args: argparse.Namespace, spark: SparkSession) -> Path:
    require(args.shuffle_partitions > 0, "shuffle partitions must be positive")
    require(args.output_partitions > 0, "output partitions must be positive")
    require(not args.output.exists(), f"Output already exists: {args.output}")
    source_manifest = read_json(args.feature_manifest.resolve())
    columns = source_columns(source_manifest)
    source = spark.read.parquet(spark_path(args.input))
    require_source_schema(source, columns)
    counts = require_labeled_source(source, source_manifest)
    labeled = source.where(F.col(SPLIT).isNotNull())
    statistics = fit_feature_statistics(labeled, columns)
    vectors = transform(labeled, statistics)
    vectors_path = args.output / "vectors.parquet"
    vectors.repartition(args.output_partitions).write.partitionBy(SPLIT).parquet(
        spark_path(vectors_path)
    )
    manifest = {
        "contract_version": OUTPUT_CONTRACT_VERSION,
        "format_version": 1,
        "source": {
            "path": args.input.as_posix(),
            "feature_manifest": args.feature_manifest.as_posix(),
            "feature_manifest_sha256": sha256_file(args.feature_manifest.resolve()),
            "feature_contract_version": source_manifest["contract_version"],
            "predictor_order_sha256": order_sha256(columns),
        },
        "counts": {
            "rows": sum(item["tracks"] for item in counts.values()),
            "splits": counts,
        },
        "target": target_contract(),
        "preprocessing": {
            "fit_split": "train",
            "imputation": "train_mean",
            "scaling": "train_sample_standard_deviation_after_imputation",
            "dimension": len(columns),
            "features": statistics,
        },
        "output": {
            "path": "vectors.parquet",
            "columns": [TRACK_ID, ARTIST_ID, YEAR, TARGET_COLUMN, FEATURES, SPLIT],
            "schema": schema_payload(vectors),
            "partition_column": SPLIT,
        },
    }
    write_json(args.output.resolve() / "manifest.json", manifest)
    print(
        "year_t90_prepared "
        f"rows={manifest['counts']['rows']}, dimension={len(columns)}, output={args.output.resolve()}"
    )
    return args.output.resolve()


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("YearPredictionPrepareT90")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        build(args, spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
