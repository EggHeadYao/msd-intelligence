from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


TRACK_ID_COLUMN = "track_id"
EMBEDDING_COLUMN = "embedding"
DEFAULT_AUDIO_DIR = Path("parquets/merlin_v2/audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the MERLIN C1 audio FAISS index.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_AUDIO_DIR / "song_embeddings_audio.parquet",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--index-name", default="index_audio.faiss")
    parser.add_argument("--track-ids-name", default="index_audio_track_ids.parquet")
    parser.add_argument("--expected-rows", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinValidateAudioFaiss")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_selected_k(output_dir: Path) -> int | None:
    metadata_path = output_dir / "audio_encoder_metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return int(metadata["selected_k"])


def validate_mapping(mapping: DataFrame, expected_rows: int) -> None:
    stats = mapping.agg(
        F.count("*").alias("rows"),
        F.countDistinct("row_id").alias("distinct_row_id"),
        F.countDistinct(TRACK_ID_COLUMN).alias("distinct_track_id"),
        F.min("row_id").alias("min_row_id"),
        F.max("row_id").alias("max_row_id"),
        F.sum(
            F.when(
                F.col("row_id").isNull() | F.col(TRACK_ID_COLUMN).isNull(),
                1,
            ).otherwise(0),
        ).alias("null_rows"),
    ).first()
    require(stats["rows"] == expected_rows, "track-id mapping row count mismatch")
    require(stats["distinct_row_id"] == expected_rows, "track-id mapping row_id is not unique")
    require(stats["distinct_track_id"] == expected_rows, "track-id mapping track_id is not unique")
    require(stats["min_row_id"] == 0, "track-id mapping must start at row_id 0")
    require(stats["max_row_id"] == expected_rows - 1, "track-id mapping max row_id mismatch")
    require(stats["null_rows"] == 0, "track-id mapping contains null rows")


def sample_queries(
    mapping: DataFrame,
    embeddings: DataFrame,
    query_count: int,
) -> list[tuple[int, str, list[float]]]:
    rows = (
        mapping.orderBy("row_id")
        .limit(query_count)
        .join(embeddings.select(TRACK_ID_COLUMN, EMBEDDING_COLUMN), TRACK_ID_COLUMN, "inner")
        .select("row_id", TRACK_ID_COLUMN, EMBEDDING_COLUMN)
        .collect()
    )
    return [(int(row["row_id"]), row[TRACK_ID_COLUMN], row[EMBEDDING_COLUMN]) for row in rows]


def validate_queries(
    index: faiss.Index,
    queries: list[tuple[int, str, list[float]]],
    top_k: int,
) -> None:
    require(queries, "no query embeddings found")
    matrix = np.vstack([np.asarray(row[2], dtype=np.float32).reshape(1, -1) for row in queries])
    require(matrix.shape[1] == index.d, "query embedding dimension does not match FAISS index")
    distances, indices = index.search(matrix, min(top_k + 1, index.ntotal))

    for query_index, (row_id, track_id, _) in enumerate(queries):
        result_ids = [int(value) for value in indices[query_index] if int(value) >= 0]
        require(result_ids, f"query {track_id} returned no FAISS results")
        require(
            all(0 <= value < index.ntotal for value in result_ids),
            f"query {track_id} returned invalid row ids",
        )
        require(row_id in result_ids, f"query {track_id} did not retrieve itself")
        non_self = [value for value in result_ids if value != row_id]
        require(
            len(non_self) >= min(top_k, index.ntotal - 1),
            f"query {track_id} has too few non-self results",
        )
        require(
            float(distances[query_index][0]) <= 1.0001,
            "inner-product score is above normalized range",
        )


def main() -> None:
    args = parse_args()
    index_path = args.output / args.index_name
    mapping_path = args.output / args.track_ids_name
    require(index_path.exists(), f"missing FAISS index: {index_path}")
    require(mapping_path.exists(), f"missing track-id mapping: {mapping_path}")

    index = faiss.read_index(str(index_path))
    selected_k = read_selected_k(args.output)
    if selected_k is not None:
        require(index.d == selected_k, "FAISS index dimension does not match selected_k")
    require(index.ntotal == args.expected_rows, "FAISS index size mismatch")

    spark = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        mapping = spark.read.parquet(spark_path(mapping_path))
        embeddings = spark.read.parquet(spark_path(args.embeddings))
        validate_mapping(mapping, args.expected_rows)
        queries = sample_queries(mapping, embeddings, args.queries)
        validate_queries(index, queries, args.top_k)
        print(
            "audio_faiss_validation_passed "
            f"rows={index.ntotal}, dimension={index.d}, queries={len(queries)}, top_k={args.top_k}",
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
