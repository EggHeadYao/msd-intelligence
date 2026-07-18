from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vector
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType

from columns import (
    CONTRACT_VERSION,
    MERLIN_ARRAY_FEATURE_COUNT,
    MERLIN_RAW_VIEW_COUNT,
    RAW_AUDIO_COLUMNS,
    SHARED_FEATURE_COUNT,
    TRACK_ID_COLUMN,
)
from preprocess import preprocess_audio_features


FEATURES_COLUMN = "features"
SCALED_FEATURES_COLUMN = "scaled_features"
PCA_FEATURES_COLUMN = "pca_features"
EMBEDDING_COLUMN = "embedding"
PCA_DIMENSION = 128
MODEL_READY_SCHEMA_VERSION = "c1_model_ready_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MERLIN C1 PCA audio encoder.")
    parser.add_argument("--input", type=Path, default=Path("parquets/prepared_v2/song_audio_features_raw.parquet"))
    parser.add_argument("--output", type=Path, default=Path("parquets/merlin_v2/audio"))
    parser.add_argument("--target-variance", type=float, default=0.95)
    parser.add_argument("--fixed-k", type=int, choices=(PCA_DIMENSION,), default=PCA_DIMENSION)
    parser.add_argument("--max-components", type=int, choices=(PCA_DIMENSION,), default=PCA_DIMENSION)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinTrainAudioPCA")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def validate_raw_input(df: DataFrame) -> int:
    missing = sorted(set(RAW_AUDIO_COLUMNS) - set(df.columns))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"C1 raw input is missing {len(missing)} columns: {preview}")

    fields = {field.name: field.dataType for field in df.schema.fields}
    if not isinstance(fields[TRACK_ID_COLUMN], StringType):
        raise TypeError("C1 track_id must use Spark string type")
    non_numeric = [
        column for column in RAW_AUDIO_COLUMNS
        if column != TRACK_ID_COLUMN and not isinstance(fields[column], NumericType)
    ]
    if non_numeric:
        preview = ", ".join(non_numeric[:10])
        raise TypeError(f"C1 raw features must be numeric: {preview}")

    numeric_values = F.array(*(
        F.col(column).cast("double") for column in RAW_AUDIO_COLUMNS
        if column != TRACK_ID_COLUMN
    ))
    has_non_finite = F.exists(
        numeric_values,
        lambda value: value.isNotNull()
        & (F.isnan(value) | (F.abs(value) == float("inf"))),
    )
    if df.where(has_non_finite).limit(1).count():
        raise ValueError("C1 raw input contains NaN or infinite feature values")

    row_count = df.count()
    if row_count == 0:
        raise ValueError("C1 raw input is empty")
    invalid_ids = df.where(
        F.col(TRACK_ID_COLUMN).isNull()
        | (~F.col(TRACK_ID_COLUMN).rlike(r"^TR.{16}$"))
    ).limit(1).count()
    if invalid_ids:
        raise ValueError("C1 raw input contains an invalid track_id")
    unique_ids = df.select(TRACK_ID_COLUMN).distinct().count()
    if unique_ids != row_count:
        raise ValueError("C1 raw input contains duplicate track_id values")
    return row_count


def cumulative(values: list[float]) -> list[float]:
    total = 0.0
    result = []
    for value in values:
        total += float(value)
        result.append(total)
    return result


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def choose_k(explained: list[float], target_variance: float, fixed_k: int) -> int:
    if fixed_k > 0:
        return min(fixed_k, len(explained))
    for index, value in enumerate(cumulative(explained), start=1):
        if value >= target_variance:
            return index
    return len(explained)


def add_normalized_embedding(df: DataFrame, k: int) -> DataFrame:
    values_col = "_embedding_values"
    norm_col = "_embedding_norm"
    values = F.slice(vector_to_array(F.col(PCA_FEATURES_COLUMN)), 1, k)
    return (
        df.withColumn(values_col, values)
        .withColumn(
            norm_col,
            F.sqrt(F.aggregate(F.col(values_col), F.lit(0.0), lambda acc, x: acc + x * x)),
        )
        .withColumn(
            EMBEDDING_COLUMN,
            F.when(
                F.col(norm_col) > 0.0,
                F.transform(F.col(values_col), lambda x: (x / F.col(norm_col)).cast("float")),
            ),
        )
        .drop(values_col, norm_col)
    )


def require_valid_embeddings(df: DataFrame) -> None:
    values = F.col(EMBEDDING_COLUMN)
    invalid = df.where(
        values.isNull()
        | F.exists(values, lambda x: x.isNull() | F.isnan(x) | (F.abs(x) == float("inf")))
    ).limit(1).count()
    if invalid:
        raise ValueError("C1 PCA produced a zero-norm or non-finite embedding")


def vector_to_list(vector: Vector) -> list[float]:
    return [float(value) for value in vector.toArray()]


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    spark = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = spark.read.parquet(spark_path(args.input))
        input_schema_hash = sha256_text(raw.schema.json())
        if args.limit > 0:
            raw = raw.limit(args.limit)
        row_count = validate_raw_input(raw)
        processed, feature_columns, preprocess_metadata = preprocess_audio_features(raw)

        assembler = VectorAssembler(inputCols=list(feature_columns), outputCol=FEATURES_COLUMN)
        assembled = assembler.transform(processed).select(TRACK_ID_COLUMN, FEATURES_COLUMN)

        scaler = StandardScaler(
            inputCol=FEATURES_COLUMN,
            outputCol=SCALED_FEATURES_COLUMN,
            withMean=True,
            withStd=True,
        )
        scaler_model = scaler.fit(assembled)
        scaled = scaler_model.transform(assembled).select(TRACK_ID_COLUMN, SCALED_FEATURES_COLUMN)

        max_components = PCA_DIMENSION
        if len(feature_columns) < PCA_DIMENSION:
            raise ValueError("C1 requires at least 128 non-constant input features")
        pca = PCA(k=max_components, inputCol=SCALED_FEATURES_COLUMN, outputCol=PCA_FEATURES_COLUMN)
        pca_model = pca.fit(scaled)
        explained = vector_to_list(pca_model.explainedVariance)
        selected_k = PCA_DIMENSION

        projected = pca_model.transform(scaled).select(TRACK_ID_COLUMN, PCA_FEATURES_COLUMN)
        embeddings = add_normalized_embedding(projected, selected_k).select(TRACK_ID_COLUMN, EMBEDDING_COLUMN)
        require_valid_embeddings(embeddings)

        args.output.mkdir(parents=True, exist_ok=True)
        embeddings.write.mode("overwrite").parquet(spark_path(args.output / "song_embeddings_audio.parquet"))
        scaler_model.write().overwrite().save(spark_path(args.output / "scaler_model"))
        pca_model.write().overwrite().save(spark_path(args.output / "pca_model"))

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "merlin_schema_version": "3.0",
            "shared_audio_contract_version": CONTRACT_VERSION,
            "c1_feature_version": 2,
            "model_ready_schema_version": MODEL_READY_SCHEMA_VERSION,
            "shared_audio_feature_count": SHARED_FEATURE_COUNT,
            "merlin_array_feature_count": MERLIN_ARRAY_FEATURE_COUNT,
            "merlin_raw_view_count": MERLIN_RAW_VIEW_COUNT,
            "input_path": str(args.input),
            "input_schema_sha256": input_schema_hash,
            "row_count": row_count,
            "feature_columns": list(feature_columns),
            "feature_order_sha256": sha256_text("\n".join(feature_columns)),
            "feature_count": len(feature_columns),
            "permanent_dropped_fields": ["danceability", "energy"],
            "embedding_column": EMBEDDING_COLUMN,
            "embedding_format": "array<float32>",
            "target_variance": args.target_variance,
            "fixed_k": args.fixed_k,
            "limit": args.limit,
            "max_components": max_components,
            "selected_k": selected_k,
            "explained_variance": explained,
            "cumulative_explained_variance": cumulative(explained),
            "preprocess": preprocess_metadata,
            "scaler_mean": vector_to_list(scaler_model.mean),
            "scaler_std": vector_to_list(scaler_model.std),
        }
        write_json(metadata, args.output / "audio_encoder_metadata.json")
        print(
            "audio_pca_training_done "
            f"rows={row_count}, features={len(feature_columns)}, "
            f"max_components={max_components}, selected_k={selected_k}, "
            f"explained={metadata['cumulative_explained_variance'][selected_k - 1]:.6f}",
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
