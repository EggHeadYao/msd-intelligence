from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from columns import INPUT_COLUMNS, SPLIT, TRAIN
from preprocessing import (
    DEFAULT_QUANTILE_ERROR,
    DEFAULT_VARIANCE_THRESHOLD,
    fit_feature_contract,
    transform_features,
    validate_binary_columns,
)
from views import build_engineered_view, build_linear_view, engineered_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only year-prediction feature views.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("parquets/year_prediction/dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("parquets/year_prediction/features"),
    )
    parser.add_argument("--feature-version", default="v1")
    parser.add_argument("--quantile-error", type=float, default=DEFAULT_QUANTILE_ERROR)
    parser.add_argument("--variance-threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    parser.add_argument("--output-partitions", type=int, default=12)
    return parser.parse_args()


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def split_counts(df: DataFrame) -> dict[str, int]:
    return {
        row[SPLIT]: int(row["count"])
        for row in df.groupBy(SPLIT).count().collect()
    }


def schema_types(df: DataFrame) -> dict[str, str]:
    return {field.name: field.dataType.simpleString() for field in df.schema.fields}


def build(args: argparse.Namespace, spark: SparkSession) -> Path:
    dataset = args.dataset.resolve()
    manifest_path = dataset / "manifest.json"
    dataset_manifest = read_json(manifest_path)
    source = spark.read.parquet(spark_path(dataset / "supervised_features.parquet"))
    if tuple(source.columns) != INPUT_COLUMNS:
        raise ValueError(f"Unexpected supervised input columns: {source.columns}")
    source.persist(StorageLevel.MEMORY_AND_DISK)
    validate_binary_columns(source)
    counts = split_counts(source)
    if counts != {
        name: int(values["tracks"])
        for name, values in dataset_manifest["counts"]["splits"].items()
    }:
        raise ValueError("Input split counts differ from the dataset manifest")

    train = source.where(F.col(SPLIT) == TRAIN)
    state = fit_feature_contract(
        train,
        quantile_error=args.quantile_error,
        variance_threshold=args.variance_threshold,
    )
    engineered = build_engineered_view(transform_features(source, state), state)
    engineered.persist(StorageLevel.MEMORY_AND_DISK)
    row_count = engineered.count()
    expected_rows = int(dataset_manifest["counts"]["labeled_tracks"])
    if row_count != expected_rows:
        raise ValueError(f"Wrong engineered row count: expected={expected_rows}, actual={row_count}")
    linear = build_linear_view(engineered, state)

    output = (args.output_root / args.feature_version).resolve()
    output.mkdir(parents=True, exist_ok=True)
    engineered.repartition(args.output_partitions, SPLIT).write.mode("overwrite").partitionBy(SPLIT).parquet(
        spark_path(output / "engineered_features.parquet")
    )
    linear.repartition(args.output_partitions, SPLIT).write.mode("overwrite").partitionBy(SPLIT).parquet(
        spark_path(output / "linear_vectors.parquet")
    )

    metadata = {
        "format_version": 1,
        "feature_version": args.feature_version,
        "source": {
            "dataset_path": str(dataset),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "artist_assignment_sha256": dataset_manifest["artist_assignment_sha256"],
        },
        "fit_scope": {"split": TRAIN, "rows": counts[TRAIN]},
        "counts": {"rows": row_count, "splits": counts},
        "target": {
            "source_column": "year",
            "output_column": "normalized_year",
            "minimum": 1922,
            "maximum": 2011,
            "formula": "(year - 1922) / 89",
        },
        "preprocessing_order": [
            "invalid_to_missing",
            "train_only_clip_and_segment_statistics",
            "clip_and_log",
            "train_only_continuous_imputation",
            "categorical_encoding",
            "train_only_variance_filter",
            "train_only_standardization",
        ],
        "preprocessing": state,
        "outputs": {
            "engineered_features": {
                "path": str(output / "engineered_features.parquet"),
                "columns": list(engineered_columns(state)),
                "schema": schema_types(engineered),
            },
            "linear_vectors": {
                "path": str(output / "linear_vectors.parquet"),
                "feature_column": "features",
                "feature_type": "array<double>",
                "dimension": len(state["retained_columns"]),
                "schema": schema_types(linear),
            },
        },
    }
    write_json(output / "preprocessing_metadata.json", metadata)
    engineered.unpersist()
    source.unpersist()
    print(
        "year_features_built "
        f"rows={row_count}, candidates={len(state['candidate_columns'])}, "
        f"retained={len(state['retained_columns'])}, output={output}"
    )
    return output


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("YearPredictionBuildFeatures")
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
