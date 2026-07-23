from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
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
    sha256_path,
    validate_c1_manifest,
    write_json_atomic,
)
from columns import CONTRACT_VERSION


TRACK_ID_COLUMN = "track_id"
EMBEDDING_COLUMN = "embedding"
DEFAULT_AUDIO_DIR = Path("parquets_new/merlin/audio")
FAISS_MANIFEST_NAME = "index_audio_manifest.json"
FAISS_MANIFEST_VERSION = "merlin_faiss_index_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the MERLIN C1 audio FAISS index."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_AUDIO_DIR / "song_embeddings_audio.parquet",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--index-name", default="index_audio.faiss")
    parser.add_argument(
        "--track-ids-name",
        default="index_audio_track_ids.parquet",
    )
    parser.add_argument("--manifest-name", default=FAISS_MANIFEST_NAME)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.limit < 0:
        raise ValueError("limit cannot be negative")
    if args.shuffle_partitions <= 0:
        raise ValueError("shuffle partitions must be positive")
    if not args.index_name:
        raise ValueError("index name cannot be empty")
    if not args.track_ids_name:
        raise ValueError("track-ids name cannot be empty")
    if not args.manifest_name:
        raise ValueError("manifest name cannot be empty")
    for option, value in (
        ("index", args.index_name),
        ("track-ids", args.track_ids_name),
        ("manifest", args.manifest_name),
    ):
        if Path(value).name != value:
            raise ValueError(f"{option} name must be a file name, not a path")
    default_names = (
        "index_audio.faiss",
        "index_audio_track_ids.parquet",
        FAISS_MANIFEST_NAME,
    )
    requested_names = (
        args.index_name,
        args.track_ids_name,
        args.manifest_name,
    )
    uses_nonproduction_names = requested_names != default_names
    if (
        args.output.resolve() == DEFAULT_AUDIO_DIR.resolve()
        and (args.limit > 0 or uses_nonproduction_names)
        and any(name in default_names for name in requested_names)
    ):
        raise ValueError(
            "test or custom FAISS builds in the production directory require "
            "non-default names for index, mapping, and manifest"
        )
    return args


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


def encoder_contract(output_dir: Path) -> tuple[int, str, int]:
    metadata_path = output_dir / "audio_encoder_metadata.json"
    require(
        metadata_path.exists(),
        f"missing C1 encoder metadata: {metadata_path}",
    )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    required = (
        "run_id",
        "shared_audio_contract_version",
        "c1_feature_version",
        "selected_k",
        "embedding_format",
        "row_count",
    )
    missing = [key for key in required if key not in metadata]
    require(
        not missing,
        f"C1 encoder metadata missing keys: {missing}",
    )

    require(
        metadata["shared_audio_contract_version"] == CONTRACT_VERSION,
        "wrong audio contract",
    )
    require(
        int(metadata["c1_feature_version"]) == 2,
        "wrong C1 feature version",
    )

    selected_k = int(metadata["selected_k"])
    require(
        selected_k > 0,
        "C1 selected PCA dimension must be positive",
    )

    max_components = metadata.get("max_components")
    if max_components is not None:
        require(
            selected_k <= int(max_components),
            "C1 selected dimension exceeds fitted PCA dimension",
        )

    require(
        metadata["embedding_format"] == "array<float32>",
        "wrong embedding format",
    )

    expected_rows = int(metadata["row_count"])
    require(expected_rows > 0, "C1 metadata row count must be positive")

    validate_c1_manifest(output_dir, metadata)
    return selected_k, str(metadata["run_id"]), expected_rows


def read_embeddings(
    spark: SparkSession,
    path: Path,
    limit: int,
) -> DataFrame:
    embeddings = spark.read.parquet(spark_path(path)).select(
        TRACK_ID_COLUMN,
        EMBEDDING_COLUMN,
    )

    if limit > 0:
        embeddings = embeddings.orderBy(TRACK_ID_COLUMN).limit(limit)
    else:
        embeddings = embeddings.orderBy(TRACK_ID_COLUMN)

    return embeddings.persist(StorageLevel.DISK_ONLY)


def write_track_id_mapping(
    embeddings: DataFrame,
    output_path: Path,
) -> None:
    schema = StructType(
        (
            StructField("row_id", LongType(), nullable=False),
            StructField(TRACK_ID_COLUMN, StringType(), nullable=False),
        )
    )

    mapping = (
        embeddings.select(TRACK_ID_COLUMN)
        .rdd.map(lambda row: row[0])
        .zipWithIndex()
        .map(lambda item: (int(item[1]), item[0]))
    )

    (
        embeddings.sparkSession.createDataFrame(mapping, schema)
        .write.mode("overwrite")
        .parquet(spark_path(output_path))
    )


def flush_batch(
    index: faiss.IndexFlatIP,
    batch: list[np.ndarray],
) -> None:
    if batch:
        index.add(np.vstack(batch))


def build_index(
    embeddings: DataFrame,
    batch_size: int,
    expected_dim: int,
) -> faiss.IndexFlatIP:
    require(batch_size > 0, "batch size must be positive")
    require(expected_dim > 0, "expected embedding dimension must be positive")

    index: faiss.IndexFlatIP | None = None
    batch: list[np.ndarray] = []
    rows = 0

    for row in embeddings.toLocalIterator():
        vector = np.asarray(row[EMBEDDING_COLUMN], dtype=np.float32)
        require(
            vector.ndim == 1,
            "embedding must be a one-dimensional array",
        )
        require(
            np.all(np.isfinite(vector)),
            "embedding contains NaN or infinite values",
        )
        require(
            vector.shape[0] == expected_dim,
            "embedding dimension does not match metadata",
        )

        norm = float(np.linalg.norm(vector))
        require(
            np.isfinite(norm) and abs(norm - 1.0) <= 1e-5,
            "embedding is not unit normalized",
        )

        if index is None:
            index = faiss.IndexFlatIP(expected_dim)
        else:
            require(
                vector.shape[0] == index.d,
                "embedding dimension changed while building index",
            )

        batch.append(vector.reshape(1, -1))
        rows += 1

        if len(batch) >= batch_size:
            flush_batch(index, batch)
            batch.clear()

        if rows % 100_000 == 0:
            print(
                f"audio_faiss_index_progress rows={rows}",
                flush=True,
            )

    require(index is not None, "no embeddings found")
    flush_batch(index, batch)
    require(
        index.ntotal == rows,
        "FAISS index size does not match processed rows",
    )
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
        print(f"audio_faiss_build_started input={args.input}, output={args.output}", flush=True)
        embeddings = read_embeddings(spark, args.input, args.limit)
        stats = embeddings.agg(
            F.count("*").alias("rows"),
            F.sum(
                F.when(
                    F.col(TRACK_ID_COLUMN).isNull() | F.col(EMBEDDING_COLUMN).isNull(),
                    1,
                ).otherwise(0),
            ).alias("null_rows"),
        ).first()
        row_count = int(stats["rows"])
        null_rows = int(stats["null_rows"] or 0)
        require(row_count > 0, "input embedding table is empty")
        require(null_rows == 0, "input embedding table contains null rows")
        print(f"audio_faiss_input_ready rows={row_count}", flush=True)

        staged_mapping = staging / args.track_ids_name
        write_track_id_mapping(embeddings, staged_mapping)
        print(f"audio_faiss_mapping_ready path={staged_mapping}", flush=True)
        index = build_index(embeddings, args.batch_size, expected_dim)
        print(f"audio_faiss_index_ready rows={index.ntotal}, dimension={index.d}", flush=True)
        staged_index = staging / args.index_name
        faiss.write_index(index, str(staged_index))
        encoder_metadata_path = args.output / "audio_encoder_metadata.json"
        manifest = {
            "artifact_type": "merlin_faiss_index",
            "manifest_version": FAISS_MANIFEST_VERSION,
            "embedding_space": "audio",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "shared_audio_contract_version": CONTRACT_VERSION,
            "c1_feature_version": 2,
            "index_type": "IndexFlatIP",
            "metric": "inner_product",
            "dimension": index.d,
            "row_count": index.ntotal,
            "index_file": args.index_name,
            "mapping_path": args.track_ids_name,
            "index_sha256": sha256_path(staged_index),
            "mapping_sha256": sha256_path(staged_mapping),
            "encoder_metadata_sha256": sha256_path(encoder_metadata_path),
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
