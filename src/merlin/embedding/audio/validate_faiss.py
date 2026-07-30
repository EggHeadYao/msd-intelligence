from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import faiss
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from merlin.embedding.audio.artifacts import (
    AUDIO_INDEX_NAME,
    AUDIO_MAPPING_NAME,
    ENCODER_METADATA_NAME,
    FAISS_MANIFEST_NAME,
    FAISS_MANIFEST_VERSION,
    load_encoder_contract,
    sha256_path,
)
from merlin.embedding.audio.columns import CONTRACT_VERSION


TRACK_ID_COLUMN = "track_id"
EMBEDDING_COLUMN = "embedding"
DEFAULT_AUDIO_DIR = Path("parquets_new/merlin/audio")
def validate_source_mapping(
    index: Any,
    queries: Sequence[tuple[int, str, Sequence[float]]],
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, float]:
    if not queries:
        raise AssertionError("no query embeddings found")
    matrix = np.vstack(
        [np.asarray(row[2], dtype=np.float32).reshape(1, -1) for row in queries]
    )
    if matrix.shape[1] != index.d:
        raise AssertionError("query embedding dimension does not match FAISS index")
    if not np.all(np.isfinite(matrix)):
        raise AssertionError("query embedding contains NaN or infinite values")
    if not np.all(np.abs(np.linalg.norm(matrix, axis=1) - 1.0) <= tolerance):
        raise AssertionError("query embedding is not unit normalized")
    reconstructed = np.vstack([index.reconstruct(row[0]) for row in queries])
    error = float(np.max(np.abs(matrix - reconstructed)))
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError("FAISS row mapping does not match source embeddings")
    return matrix, error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the MERLIN C1 audio FAISS index.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_AUDIO_DIR / "song_embeddings_audio.parquet",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--index-name", default=AUDIO_INDEX_NAME)
    parser.add_argument("--track-ids-name", default=AUDIO_MAPPING_NAME)
    parser.add_argument("--manifest-name", default=FAISS_MANIFEST_NAME)
    parser.add_argument("--expected-rows", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument(
        "--allow-noncanonical-dimension",
        action="store_true",
        help="Allow an isolated smoke index whose encoder dimension is not 128.",
    )
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


def validate_index_runtime(index: faiss.Index) -> None:
    require(type(index).__name__ == "IndexFlatIP", "FAISS runtime index is not IndexFlatIP")
    require(
        index.metric_type == faiss.METRIC_INNER_PRODUCT,
        "FAISS runtime metric is not inner product",
    )


def validate_manifest(
    output_dir: Path,
    index_path: Path,
    mapping_path: Path,
    manifest_name: str,
    expected_rows: int,
    expected_dimension: int,
    encoder_run_id: str,
    encoder_rows: int,
) -> dict[str, Any]:
    manifest_path = output_dir / manifest_name
    metadata_path = output_dir / ENCODER_METADATA_NAME
    require(manifest_path.exists(), f"missing FAISS manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    require(
        manifest.get("artifact_type") == "merlin_faiss_index",
        "wrong FAISS artifact type",
    )
    require(
        manifest.get("manifest_version") == FAISS_MANIFEST_VERSION,
        "wrong manifest version",
    )
    require(manifest.get("embedding_space") == "audio", "wrong embedding space")
    require(
        manifest.get("shared_audio_contract_version") == CONTRACT_VERSION,
        "wrong manifest contract",
    )
    require(int(manifest.get("c1_feature_version", -1)) == 2, "wrong manifest feature version")
    require(manifest.get("index_type") == "IndexFlatIP", "wrong FAISS index type")
    require(manifest.get("metric") == "inner_product", "wrong FAISS metric")
    require(
        int(manifest.get("dimension", -1)) == expected_dimension,
        "wrong manifest dimension",
    )
    require(int(manifest.get("row_count", -1)) == expected_rows, "wrong manifest row count")
    require(manifest.get("index_file") == index_path.name, "wrong manifest index path")
    require(manifest.get("mapping_path") == mapping_path.name, "wrong manifest mapping path")
    require(manifest.get("index_sha256") == sha256_path(index_path), "wrong index hash")
    require(manifest.get("mapping_sha256") == sha256_path(mapping_path), "wrong mapping hash")
    require(
        manifest.get("encoder_metadata_sha256") == sha256_path(metadata_path),
        "wrong encoder metadata hash",
    )
    require(manifest.get("encoder_run_id") == encoder_run_id, "encoder run mismatch")
    partial = manifest.get("partial_index") is True
    requested_limit = int(manifest.get("requested_limit", -1))
    if partial:
        require(requested_limit > 0, "partial FAISS manifest has no positive limit")
        require(expected_rows <= min(requested_limit, encoder_rows), "partial FAISS row count mismatch")
    else:
        require(requested_limit == 0, "complete FAISS manifest has a requested limit")
        require(expected_rows == encoder_rows, "complete FAISS index does not cover the encoder")
    return manifest


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
) -> float:
    matrix, reconstruction_error = validate_source_mapping(index, queries)
    distances, indices = index.search(matrix, min(top_k + 1, index.ntotal))
    compared_pairs = 0
    max_score_error = 0.0

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
        for rank, result_id in enumerate(result_ids):
            expected = float(np.dot(matrix[query_index], index.reconstruct(result_id)))
            error = abs(expected - float(distances[query_index][rank]))
            max_score_error = max(max_score_error, error)
            compared_pairs += 1
    require(compared_pairs >= 100, "FAISS validation requires at least 100 score comparisons")
    require(max_score_error <= 1e-5, "FAISS scores disagree with NumPy inner products")
    return reconstruction_error


def main() -> None:
    args = parse_args()
    require(args.queries > 0, "query count must be positive")
    require(args.top_k > 0, "top-k must be positive")
    require(args.expected_rows > 0, "expected rows must be positive")
    for name in (args.index_name, args.track_ids_name, args.manifest_name):
        require(Path(name).name == name and bool(name), "FAISS artifact names must be file names")
    index_path = args.output / args.index_name
    mapping_path = args.output / args.track_ids_name
    require(index_path.exists(), f"missing FAISS index: {index_path}")
    require(mapping_path.exists(), f"missing track-id mapping: {mapping_path}")
    _metadata, encoder = load_encoder_contract(
        args.output,
        require_canonical_dimension=not args.allow_noncanonical_dimension,
    )
    validate_manifest(
        args.output,
        index_path,
        mapping_path,
        args.manifest_name,
        args.expected_rows,
        encoder.selected_k,
        encoder.run_id,
        encoder.row_count,
    )

    index = faiss.read_index(str(index_path))
    validate_index_runtime(index)
    require(index.d == encoder.selected_k, "FAISS index dimension does not match selected_k")
    require(index.ntotal == args.expected_rows, "FAISS index size mismatch")

    spark = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        mapping = spark.read.parquet(spark_path(mapping_path))
        embeddings = spark.read.parquet(spark_path(args.embeddings))
        validate_mapping(mapping, args.expected_rows)
        queries = sample_queries(mapping, embeddings, args.queries)
        reconstruction_error = validate_queries(index, queries, args.top_k)
        print(
            "audio_faiss_validation_passed "
            f"rows={index.ntotal}, dimension={index.d}, queries={len(queries)}, "
            f"top_k={args.top_k}, reconstruction_error={reconstruction_error:.3g}",
        )
    finally:
        try:
            spark.stop()
        except Exception as error:
            warnings.warn(f"failed to stop Spark cleanly: {error}", RuntimeWarning)


if __name__ == "__main__":
    main()
