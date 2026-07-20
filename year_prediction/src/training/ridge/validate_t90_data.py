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


def validate_output(
    vectors: DataFrame,
    manifest: dict[str, Any],
    dimension: int,
) -> dict[str, dict[str, int]]:
    require(tuple(vectors.columns) == OUTPUT_COLUMNS, "Vector column order differs")
    expected_types = {
        TRACK_ID: "string",
        ARTIST_ID: "string",
        YEAR: "int",
        TARGET_COLUMN: "double",
        FEATURES: "array<double>",
        SPLIT: "string",
    }
    actual_types = {field.name: field.dataType.simpleString() for field in vectors.schema.fields}
    require(actual_types == expected_types, "Vector schema differs")
    require(schema_payload(vectors) == manifest["output"]["schema"], "Manifest schema differs")
    target = target_contract()
    expected_label = (F.col(YEAR).cast("double") - target["minimum"]) / target["span"]
    invalid = (
        F.col(TRACK_ID).isNull()
        | (F.col(TRACK_ID) == "")
        | F.col(ARTIST_ID).isNull()
        | (F.col(ARTIST_ID) == "")
        | F.col(YEAR).isNull()
        | ~F.col(YEAR).between(target["minimum"], target["maximum"])
        | ~F.col(SPLIT).isin(*SPLITS)
        | F.col(TARGET_COLUMN).isNull()
        | F.isnan(TARGET_COLUMN)
        | (F.abs(F.col(TARGET_COLUMN)) == F.lit(float("inf")))
        | (F.abs(F.col(TARGET_COLUMN) - expected_label) > VALUE_TOLERANCE)
        | F.col(FEATURES).isNull()
        | (F.size(FEATURES) != dimension)
        | ~finite_array(F.col(FEATURES))
    )
    rows = vectors.groupBy(SPLIT).agg(
        F.count("*").alias("tracks"),
        F.countDistinct(TRACK_ID).alias("distinct_tracks"),
        F.countDistinct(ARTIST_ID).alias("artists"),
        F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid"),
    ).collect()
    counts = {
        row[SPLIT]: {"tracks": int(row["tracks"]), "artists": int(row["artists"])}
        for row in rows
    }
    expected_counts = {
        name: {"tracks": int(item["tracks"]), "artists": int(item["artists"])}
        for name, item in manifest["counts"]["splits"].items()
    }
    require(counts == expected_counts, "Vector split counts differ")
    require(
        sum(item["tracks"] for item in counts.values()) == int(manifest["counts"]["rows"]),
        "Vector row count differs",
    )
    require(
        all(int(row["tracks"]) == int(row["distinct_tracks"]) for row in rows),
        "Vector track IDs are duplicated",
    )
    require(all(int(row["invalid"]) == 0 for row in rows), "Vector output contains invalid rows")
    require(
        vectors.select(TRACK_ID).distinct().count() == int(manifest["counts"]["rows"]),
        "Vector track IDs overlap across splits",
    )
    overlap = (
        vectors.groupBy(ARTIST_ID)
        .agg(F.countDistinct(SPLIT).alias("split_count"))
        .where(F.col("split_count") > 1)
        .limit(1)
        .count()
    )
    require(overlap == 0, "Vector artists overlap across splits")
    return counts


def recompute_statistics(
    source: DataFrame,
    statistics: list[dict[str, Any]],
) -> None:
    names = [str(item["name"]) for item in statistics]
    train = source.where(F.col(SPLIT) == "train")
    mean_row = train.agg(
        F.count("*").alias("_rows"),
        *(F.count(name).alias(f"_count_{index}") for index, name in enumerate(names)),
        *(F.avg(name).alias(f"_mean_{index}") for index, name in enumerate(names)),
    ).first()
    train_count = int(mean_row["_rows"])
    means: list[float] = []
    for index, item in enumerate(statistics):
        mean = float(mean_row[f"_mean_{index}"])
        count = int(mean_row[f"_count_{index}"])
        require(
            count == int(item["finite_train_count"]),
            f"Finite count differs for {item['name']}",
        )
        require(
            train_count - count == int(item["imputed_train_count"]),
            f"Missing count differs for {item['name']}",
        )
        require(
            math.isclose(
                mean,
                float(item["mean"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ),
            f"Train mean differs for {item['name']}",
        )
        means.append(mean)
    imputed = train.select(
        *(F.coalesce(F.col(name), F.lit(mean)).alias(name) for name, mean in zip(names, means))
    )
    std_row = imputed.agg(
        *(F.stddev_samp(name).alias(f"_std_{index}") for index, name in enumerate(names))
    ).first()
    for index, item in enumerate(statistics):
        require(
            math.isclose(
                float(std_row[f"_std_{index}"]),
                float(item["standard_deviation"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ),
            f"Train standard deviation differs for {item['name']}",
        )


def require_source_equivalence(
    source: DataFrame,
    vectors: DataFrame,
    statistics: list[dict[str, Any]],
) -> None:
    feature_values = [
        (
            (F.coalesce(F.col(str(item["name"])), F.lit(float(item["mean"]))) - float(item["mean"]))
            / float(item["standard_deviation"])
        ).cast("double")
        for item in statistics
    ]
    target = target_contract()
    expected = source.where(F.col(SPLIT).isNotNull()).select(
        TRACK_ID,
        ARTIST_ID,
        YEAR,
        SPLIT,
        ((F.col(YEAR).cast("double") - target["minimum"]) / target["span"]).alias(TARGET_COLUMN),
        F.array(*feature_values).alias(FEATURES),
    )
    joined = expected.alias("expected").join(vectors.alias("actual"), TRACK_ID, "full_outer")
    vector_match = F.forall(
        F.zip_with(
            F.col(f"expected.{FEATURES}"),
            F.col(f"actual.{FEATURES}"),
            lambda left, right: F.abs(left - right) <= F.lit(VALUE_TOLERANCE),
        ),
        lambda value: value,
    )
    matches = (
        F.col(f"expected.{TRACK_ID}").isNotNull()
        & F.col(f"actual.{TRACK_ID}").isNotNull()
        & F.col(f"expected.{ARTIST_ID}").eqNullSafe(F.col(f"actual.{ARTIST_ID}"))
        & F.col(f"expected.{YEAR}").eqNullSafe(F.col(f"actual.{YEAR}"))
        & F.col(f"expected.{SPLIT}").eqNullSafe(F.col(f"actual.{SPLIT}"))
        & (
            F.abs(
                F.col(f"expected.{TARGET_COLUMN}") - F.col(f"actual.{TARGET_COLUMN}")
            )
            <= VALUE_TOLERANCE
        )
        & vector_match
    )
    require(
        joined.where(~F.coalesce(matches, F.lit(False))).limit(1).count() == 0,
        "Vector contents differ from the source transformation",
    )


