from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType


TRACK_ID_COLUMN = "track_id"
EMBEDDING_COLUMN = "embedding"
DEFAULT_AUDIO_DIR = Path("parquets/merlin_v2/audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the MERLIN C1 audio FAISS index.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_AUDIO_DIR / "song_embeddings_audio.parquet",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--index-name", default="index_audio.faiss")
    parser.add_argument("--track-ids-name", default="index_audio_track_ids.parquet")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinBuildAudioFaiss")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expected_dimension(output_dir: Path) -> int | None:
    metadata_path = output_dir / "audio_encoder_metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return int(metadata["selected_k"])


def read_embeddings(spark: SparkSession, path: Path, limit: int) -> DataFrame:
    embeddings = spark.read.parquet(spark_path(path)).select(TRACK_ID_COLUMN, EMBEDDING_COLUMN)
    if limit > 0:
        embeddings = embeddings.orderBy(TRACK_ID_COLUMN).limit(limit)
    else:
        embeddings = embeddings.orderBy(TRACK_ID_COLUMN)
    return embeddings.persist(StorageLevel.DISK_ONLY)


def write_track_id_mapping(embeddings: DataFrame, output_path: Path) -> None:
    schema = StructType(
        (
            StructField("row_id", LongType(), nullable=False),
            StructField(TRACK_ID_COLUMN, StringType(), nullable=False),
        ),
    )
    mapping = embeddings.select(TRACK_ID_COLUMN).rdd.map(lambda row: row[0]).zipWithIndex()
    mapping = mapping.map(lambda item: (int(item[1]), item[0]))
    embeddings.sparkSession.createDataFrame(mapping, schema).write.mode("overwrite").parquet(
        spark_path(output_path),
    )


def flush_batch(index: faiss.IndexFlatIP, batch: list[np.ndarray]) -> None:
    if batch:
        index.add(np.vstack(batch))


def build_index(
    embeddings: DataFrame,
    batch_size: int,
    expected_dim: int | None,
) -> faiss.IndexFlatIP:
    require(batch_size > 0, "batch size must be positive")
    index: faiss.IndexFlatIP | None = None
    batch: list[np.ndarray] = []
    rows = 0

    for row in embeddings.toLocalIterator():
        vector = np.asarray(row[EMBEDDING_COLUMN], dtype=np.float32)
        require(vector.ndim == 1, "embedding must be a one-dimensional array")
        require(np.all(np.isfinite(vector)), "embedding contains NaN or infinite values")
        norm = float(np.linalg.norm(vector))
        require(np.isfinite(norm) and abs(norm - 1.0) <= 1e-5, "embedding is not unit normalized")
        if expected_dim is not None:
            require(vector.shape[0] == expected_dim, "embedding dimension does not match metadata")
        if index is None:
            index = faiss.IndexFlatIP(int(vector.shape[0]))
        else:
            require(vector.shape[0] == index.d, "embedding dimension changed while building index")
        batch.append(vector.reshape(1, -1))
        rows += 1
        if len(batch) >= batch_size:
            flush_batch(index, batch)
            batch.clear()

    require(index is not None, "no embeddings found")
    flush_batch(index, batch)
    require(index.ntotal == rows, "FAISS index size does not match processed rows")
    return index


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected_dim = expected_dimension(args.output)

    spark = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        embeddings = read_embeddings(spark, args.input, args.limit)
        row_count = embeddings.count()
        null_rows = embeddings.where(
            F.col(TRACK_ID_COLUMN).isNull() | F.col(EMBEDDING_COLUMN).isNull(),
        ).count()
        require(row_count > 0, "input embedding table is empty")
        require(null_rows == 0, "input embedding table contains null rows")

        write_track_id_mapping(embeddings, args.output / args.track_ids_name)
        index = build_index(embeddings, args.batch_size, expected_dim)
        faiss.write_index(index, str(args.output / args.index_name))
        print(
            "audio_faiss_build_done "
            f"rows={row_count}, dimension={index.d}, index_size={index.ntotal}, "
            f"index={args.output / args.index_name}, track_ids={args.output / args.track_ids_name}",
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
