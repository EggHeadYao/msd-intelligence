from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vector
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from columns import TRACK_ID_COLUMN
from preprocess import preprocess_audio_features


FEATURES_COLUMN = "features"
SCALED_FEATURES_COLUMN = "scaled_features"
PCA_FEATURES_COLUMN = "pca_features"
EMBEDDING_COLUMN = "embedding"
PCA_DIMENSION = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MERLIN C1 PCA audio encoder.")
    parser.add_argument("--input", type=Path, default=Path("parquets/prepared/song_audio_features_raw.parquet"))
    parser.add_argument("--output", type=Path, default=Path("parquets/merlin/audio"))
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


def cumulative(values: list[float]) -> list[float]:
    total = 0.0
    result = []
    for value in values:
        total += float(value)
        result.append(total)
    return result


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
            F.transform(
                F.col(values_col),
                lambda x: F.when(F.col(norm_col) > 0.0, x / F.col(norm_col)).otherwise(F.lit(0.0)),
            ),
        )
        .drop(values_col, norm_col)
    )


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
        if args.limit > 0:
            raw = raw.limit(args.limit)
        row_count = raw.count()
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

        args.output.mkdir(parents=True, exist_ok=True)
        embeddings.write.mode("overwrite").parquet(spark_path(args.output / "song_embeddings_audio.parquet"))
        scaler_model.write().overwrite().save(spark_path(args.output / "scaler_model"))
        pca_model.write().overwrite().save(spark_path(args.output / "pca_model"))

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": str(args.input),
            "row_count": row_count,
            "feature_columns": list(feature_columns),
            "feature_count": len(feature_columns),
            "embedding_column": EMBEDDING_COLUMN,
            "embedding_format": "array<double>",
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
