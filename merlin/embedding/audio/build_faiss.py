from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType

from artifacts import (
    remove_path,
    replace_artifact,
    validate_c1_manifest,
    write_json_atomic,
)
from shared_contract import CONTRACT_VERSION


TRACK_ID_COLUMN = "track_id"
EMBEDDING_COLUMN = "embedding"
DEFAULT_AUDIO_DIR = Path("parquets_new/merlin/audio")
FAISS_MANIFEST_NAME = "index_audio_manifest.json"


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


def encoder_contract(output_dir: Path) -> tuple[int, str]:
    metadata_path = output_dir / "audio_encoder_metadata.json"
    require(metadata_path.exists(), f"missing C1 encoder metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = ("shared_audio_contract_version", "c1_feature_version", "selected_k", "embedding_format")
    missing = [key for key in required if key not in metadata]
    require(not missing, f"C1 encoder metadata missing keys: {missing}")
    require(metadata["shared_audio_contract_version"] == CONTRACT_VERSION, "wrong audio contract")
    require(int(metadata["c1_feature_version"]) == 2, "wrong C1 feature version")
    require(int(metadata["selected_k"]) == 128, "C1 FAISS dimension must be 128")
    require(metadata["embedding_format"] == "array<float32>", "wrong embedding format")
    validate_c1_manifest(output_dir, metadata)
    return 128, str(metadata["run_id"])


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
    expected_dim: int,
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
    run_id = str(uuid4())
    args.output.mkdir(parents=True, exist_ok=True)
    expected_dim, encoder_run_id = encoder_contract(args.output)

    spark = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    embeddings: DataFrame | None = None
    staging: Path | None = None
    try:
        staging = args.output / f".faiss-staging-{run_id}"
        staging.mkdir()
        embeddings = read_embeddings(spark, args.input, args.limit)
        row_count = embeddings.count()
        null_rows = embeddings.where(
            F.col(TRACK_ID_COLUMN).isNull() | F.col(EMBEDDING_COLUMN).isNull(),
        ).count()
        require(row_count > 0, "input embedding table is empty")
        require(null_rows == 0, "input embedding table contains null rows")

        staged_mapping = staging / args.track_ids_name
        write_track_id_mapping(embeddings, staged_mapping)
        index = build_index(embeddings, args.batch_size, expected_dim)
        staged_index = staging / args.index_name
        faiss.write_index(index, str(staged_index))
        manifest = {
            "shared_audio_contract_version": CONTRACT_VERSION,
            "c1_feature_version": 2,
            "index_type": "IndexFlatIP",
            "dimension": index.d,
            "row_count": index.ntotal,
            "index_file": args.index_name,
            "track_ids_path": args.track_ids_name,
            "encoder_run_id": encoder_run_id,
        }
        staged_manifest = staging / FAISS_MANIFEST_NAME
        write_json_atomic(manifest, staged_manifest)
        published_manifest = args.output / FAISS_MANIFEST_NAME
        published_manifest.unlink(missing_ok=True)
        replace_artifact(staged_mapping, args.output / args.track_ids_name, run_id)
        replace_artifact(staged_index, args.output / args.index_name, run_id)
        staged_manifest.replace(published_manifest)
        print(
            "audio_faiss_build_done "
            f"rows={row_count}, dimension={index.d}, index_size={index.ntotal}, "
            f"index={args.output / args.index_name}, "
            f"track_ids={args.output / args.track_ids_name}",
        )
    finally:
        if embeddings is not None:
            try:
                embeddings.unpersist(blocking=False)
            except Exception as error:
                warnings.warn(f"failed to unpersist FAISS input: {error}", RuntimeWarning)
        try:
            spark.stop()
        except Exception as error:
            warnings.warn(f"failed to stop Spark cleanly: {error}", RuntimeWarning)
        if staging is not None and staging.exists():
            try:
                remove_path(staging)
            except OSError as error:
                warnings.warn(f"failed to remove FAISS staging: {error}", RuntimeWarning)


if __name__ == "__main__":
    main()
