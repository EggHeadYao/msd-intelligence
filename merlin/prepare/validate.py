# ruff: noqa: T201
"""Fail-closed validation for canonical MERLIN prepared tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from merlin.artifacts.contract import (
    OUTPUT_MARKER_NAME,
    read_manifest,
    sha256_file,
    utc_now,
    write_manifest,
)
from merlin.prepare.contract import (
    ARTIFACT_TYPE,
    ARTIFACT_VERSION,
    AUDIO_COLUMNS,
    EDGE_TYPES,
    FEATURE_CONTRACT_NAME,
    GRAPH_EDGE_COLUMNS,
    MANIFEST_NAME,
    MERLIN_AUDIO_FEATURE_COUNT,
    MERLIN_RAW_FEATURE_COUNT,
    METADATA_COLUMNS,
    NODE_TYPES,
    OUTPUT_DIRS,
    SHARED_AUDIO_CONTRACT_VERSION,
    SHARED_AUDIO_FEATURE_COUNT,
    SUMMARY_AUDIO_COLUMNS,
    ExpectedCounts,
)
from merlin.prepare.prepare import _schema_hash, spark_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate canonical MERLIN prepared tables.",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("../parquets_new/prepared"),
        help="Prepared MERLIN directory.",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=32,
        help="Spark SQL shuffle partitions.",
    )
    parser.add_argument("--expected-songs", type=int, default=1_000_000)
    parser.add_argument("--expected-track-release", type=int, default=999_997)
    parser.add_argument("--expected-artist-term", type=int, default=1_109_381)
    parser.add_argument(
        "--expected-artist-similarity",
        type=int,
        default=2_201_916,
    )
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinValidate")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_outputs(spark: SparkSession, prepared_dir: Path) -> dict[str, DataFrame]:
    return {
        "songs_metadata": spark.read.parquet(
            spark_path(prepared_dir / "songs_metadata.parquet"),
        ),
        "song_audio_features": spark.read.parquet(
            spark_path(prepared_dir / "song_audio_features_raw.parquet"),
        ),
        "graph_edges": spark.read.parquet(
            spark_path(prepared_dir / "graph_edges.parquet"),
        ),
    }


def validate_output_layout(prepared_dir: Path) -> None:
    require(prepared_dir.is_dir(), f"prepared directory does not exist: {prepared_dir}")
    actual_dirs: set[str] = {
        path.name for path in prepared_dir.iterdir() if path.is_dir()
    }
    missing: set[str] = set(OUTPUT_DIRS) - actual_dirs
    extra: set[str] = actual_dirs - set(OUTPUT_DIRS)
    require(not missing, f"prepared directory missing outputs: {sorted(missing)}")
    require(not extra, f"prepared directory has unexpected outputs: {sorted(extra)}")
    require(
        (prepared_dir / OUTPUT_MARKER_NAME).is_file(),
        f"prepared directory is missing {OUTPUT_MARKER_NAME}",
    )
    require(
        (prepared_dir / MANIFEST_NAME).is_file(),
        f"prepared directory is missing {MANIFEST_NAME}",
    )


def require_exact_columns(
    frame: DataFrame,
    expected: tuple[str, ...],
    table_name: str,
) -> None:
    actual: tuple[str, ...] = tuple(frame.columns)
    require(
        actual == expected,
        f"{table_name} columns mismatch: expected={expected}, actual={actual}",
    )


def require_type(
    frame: DataFrame,
    table_name: str,
    column: str,
    allowed: tuple[type[T.DataType], ...],
) -> None:
    data_type: T.DataType = frame.schema[column].dataType
    require(
        isinstance(data_type, allowed),
        f"{table_name}.{column} has type {data_type}, expected {allowed}",
    )


def validate_schema_contract(tables: dict[str, DataFrame]) -> None:
    metadata: DataFrame = tables["songs_metadata"]
    audio: DataFrame = tables["song_audio_features"]
    graph_edges: DataFrame = tables["graph_edges"]

    require_exact_columns(metadata, METADATA_COLUMNS, "songs_metadata")
    require_exact_columns(audio, AUDIO_COLUMNS, "song_audio_features_raw")
    require_exact_columns(graph_edges, GRAPH_EDGE_COLUMNS, "graph_edges")

    metadata_strings: tuple[str, ...] = (
        "track_id",
        "song_id",
        "title",
        "artist_id",
        "artist_name",
        "artist_mbid",
        "release",
    )
    for column in metadata_strings:
        require_type(metadata, "songs_metadata", column, (T.StringType,))
    for column in set(METADATA_COLUMNS) - set(metadata_strings):
        require_type(metadata, "songs_metadata", column, (T.NumericType,))

    require_type(audio, "song_audio_features_raw", "track_id", (T.StringType,))
    for column in AUDIO_COLUMNS[1:]:
        require_type(audio, "song_audio_features_raw", column, (T.NumericType,))

    for column in ("src_type", "src_id", "dst_type", "dst_id", "edge_type"):
        require_type(graph_edges, "graph_edges", column, (T.StringType,))
    require_type(graph_edges, "graph_edges", "directed", (T.BooleanType,))


def _count_distinct(frame: DataFrame, columns: tuple[str, ...]) -> int:
    return frame.select(*columns).distinct().count()


def _has_invalid_numeric(
    frame: DataFrame,
    columns: tuple[str, ...],
    *,
    allow_null: bool = False,
    chunk_size: int = 48,
) -> bool:
    for start in range(0, len(columns), chunk_size):
        condition = F.lit(False)
        for column in columns[start : start + chunk_size]:
            value = F.col(column).cast("double")
            invalid = F.isnan(value) | value.isin(float("inf"), float("-inf"))
            if not allow_null:
                invalid = invalid | value.isNull()
            condition = condition | invalid
        if frame.where(condition).limit(1).count():
            return True
    return False


def validate_song_tables(
    tables: dict[str, DataFrame],
    expected: ExpectedCounts,
) -> dict[str, int]:
    metadata: DataFrame = tables["songs_metadata"]
    audio: DataFrame = tables["song_audio_features"]

    metadata_count: int = metadata.count()
    metadata_distinct: int = _count_distinct(metadata, ("track_id",))
    audio_count: int = audio.count()
    audio_distinct: int = _count_distinct(audio, ("track_id",))
    require(metadata_count == expected.songs, "songs_metadata row count mismatch")
    require(metadata_distinct == expected.songs, "songs_metadata track IDs not unique")
    require(audio_count == expected.songs, "song_audio_features row count mismatch")
    require(audio_distinct == expected.songs, "audio track IDs not unique")

    missing_audio: int = (
        metadata.select("track_id")
        .join(
            audio.select("track_id"),
            "track_id",
            "left_anti",
        )
        .count()
    )
    missing_metadata: int = (
        audio.select("track_id")
        .join(
            metadata.select("track_id"),
            "track_id",
            "left_anti",
        )
        .count()
    )
    require(missing_audio == 0, "metadata contains track IDs missing from audio")
    require(missing_metadata == 0, "audio contains track IDs missing from metadata")

    required_metadata: tuple[str, ...] = (
        "track_id",
        "song_id",
        "artist_id",
        "track_7digitalid",
        "duration",
    )
    invalid_metadata = F.lit(False)
    for column in required_metadata:
        invalid_metadata = invalid_metadata | F.col(column).isNull()
    invalid_metadata = invalid_metadata | (F.col("track_id") == "")
    invalid_metadata = invalid_metadata | (F.col("song_id") == "")
    invalid_metadata = invalid_metadata | (F.col("artist_id") == "")
    require(
        metadata.where(invalid_metadata).limit(1).count() == 0,
        "songs_metadata contains invalid required fields",
    )

    invalid_has_year: int = (
        metadata.where(
            F.col("has_year").isNull() | ~F.col("has_year").isin(0, 1),
        )
        .limit(1)
        .count()
    )
    require(invalid_has_year == 0, "has_year contains values outside {0,1}")

    release_count: int = metadata.where(
        F.col("release_7digitalid").isNotNull() & (F.col("release_7digitalid") > 0),
    ).count()
    invalid_release: int = (
        metadata.where(
            F.col("release_7digitalid").isNotNull()
            & (F.col("release_7digitalid") <= 0),
        )
        .limit(1)
        .count()
    )
    require(release_count == expected.track_release, "valid release ID count mismatch")
    require(invalid_release == 0, "release_7digitalid contains non-positive IDs")

    nullable_summary: frozenset[str] = frozenset({"tempo", "time_signature"})
    required_summary: tuple[str, ...] = tuple(
        column for column in SUMMARY_AUDIO_COLUMNS[1:] if column not in nullable_summary
    )
    require(
        not _has_invalid_numeric(audio, required_summary),
        "required Summary audio fields contain null or NaN values",
    )
    for column in nullable_summary:
        invalid_optional: int = (
            audio.where(
                F.col(column).isNotNull()
                & (F.isnan(F.col(column).cast("double")) | (F.col(column) <= 0)),
            )
            .limit(1)
            .count()
        )
        require(invalid_optional == 0, f"{column} contains an invalid observed value")

    invalid_summary_semantics: int = (
        audio.where(
            (F.col("duration") <= 0)
            | ~F.col("key").isin(*range(12))
            | ~F.col("mode").isin(0, 1)
            | (F.col("end_of_fade_in") < 0)
            | (F.col("start_of_fade_out") < F.col("end_of_fade_in"))
            | (F.col("key_confidence") < 0)
            | (F.col("key_confidence") > 1)
            | (F.col("mode_confidence") < 0)
            | (F.col("mode_confidence") > 1)
            | (F.col("time_signature_confidence") < 0)
            | (F.col("time_signature_confidence") > 1),
        )
        .limit(1)
        .count()
    )
    require(invalid_summary_semantics == 0, "Summary audio semantics are invalid")

    extracted_features: tuple[str, ...] = AUDIO_COLUMNS[len(SUMMARY_AUDIO_COLUMNS) :]
    require(
        not _has_invalid_numeric(audio, extracted_features, allow_null=True),
        "extracted audio features contain NaN or Inf values",
    )
    bounded_features: tuple[str, ...] = tuple(
        column
        for column in extracted_features
        if column.endswith("_confidence_mean")
        or column.endswith("_low_confidence_fraction")
        or column
        in {
            "invalid_segment_duration_fraction",
            "valid_analysis_duration_ratio",
        }
    )
    bounded_violation = F.lit(False)
    for column in bounded_features:
        bounded_violation = (
            bounded_violation | (F.col(column) < -1e-9) | (F.col(column) > 1.0 + 1e-9)
        )
    require(
        audio.where(bounded_violation).limit(1).count() == 0,
        "audio confidence/fraction features contain values outside [0,1]",
    )
    mask_columns: tuple[str, ...] = tuple(
        column for column in AUDIO_COLUMNS if column.startswith("has_")
    )
    for column in mask_columns:
        invalid_mask: int = (
            audio.where(
                F.col(column).isNull() | ~F.col(column).isin(0.0, 1.0),
            )
            .limit(1)
            .count()
        )
        require(invalid_mask == 0, f"{column} contains values outside {{0,1}}")

    print(
        "song_tables "
        f"metadata={metadata_count}, audio={audio_count}, valid_releases={release_count}",
    )
    return {
        "songs_metadata": metadata_count,
        "song_audio_features_raw": audio_count,
        "valid_release_ids": release_count,
    }


def _relation_violation(
    graph_edges: DataFrame,
    edge_type: str,
    src_type: str,
    dst_type: str,
    directed: bool,
) -> int:
    return (
        graph_edges.where(
            (F.col("edge_type") == edge_type)
            & (
                (F.col("src_type") != src_type)
                | (F.col("dst_type") != dst_type)
                | (F.col("directed") != F.lit(directed))
            ),
        )
        .limit(1)
        .count()
    )


def _require_same_pairs(
    observed: DataFrame,
    expected: DataFrame,
    message: str,
) -> None:
    pair_columns: list[str] = ["src_id", "dst_id"]
    missing: int = expected.join(observed, pair_columns, "left_anti").limit(1).count()
    extra: int = observed.join(expected, pair_columns, "left_anti").limit(1).count()
    require(missing == 0 and extra == 0, message)


def validate_graph_edges(
    tables: dict[str, DataFrame],
    expected: ExpectedCounts,
) -> dict[str, int]:
    metadata: DataFrame = tables["songs_metadata"]
    graph_edges: DataFrame = tables["graph_edges"]
    counts: dict[str, int] = {
        row["edge_type"]: row["count"]
        for row in graph_edges.groupBy("edge_type").count().collect()
    }
    require(set(counts) == set(EDGE_TYPES), f"unexpected graph edge types: {counts}")
    expected_counts: dict[str, int] = {
        "track_artist": expected.songs,
        "track_release": expected.track_release,
        "artist_term": expected.artist_term,
        "artist_similarity": expected.artist_similarity,
    }
    require(counts == expected_counts, f"graph edge counts mismatch: {counts}")
    require(
        sum(counts.values()) == expected.graph_edges, "graph total row count mismatch"
    )

    null_or_empty = F.lit(False)
    for column in ("src_type", "src_id", "dst_type", "dst_id", "edge_type"):
        null_or_empty = null_or_empty | F.col(column).isNull() | (F.col(column) == "")
    null_or_empty = null_or_empty | F.col("directed").isNull()
    require(
        graph_edges.where(null_or_empty).limit(1).count() == 0,
        "graph_edges contains null or empty fields",
    )
    require(
        graph_edges.where(
            ~F.col("src_type").isin(*NODE_TYPES) | ~F.col("dst_type").isin(*NODE_TYPES),
        )
        .limit(1)
        .count()
        == 0,
        "graph_edges contains unexpected node types",
    )
    require(
        _count_distinct(graph_edges, ("edge_type", "src_id", "dst_id"))
        == expected.graph_edges,
        "graph_edges contains duplicate typed pairs",
    )

    relation_specs: tuple[tuple[str, str, str, bool], ...] = (
        ("track_artist", "track", "artist", False),
        ("track_release", "track", "release", False),
        ("artist_term", "artist", "term", False),
        ("artist_similarity", "artist", "artist", True),
    )
    for edge_type, src_type, dst_type, directed in relation_specs:
        require(
            _relation_violation(
                graph_edges,
                edge_type,
                src_type,
                dst_type,
                directed,
            )
            == 0,
            f"{edge_type} relation semantics mismatch",
        )

    observed_artist: DataFrame = graph_edges.where(
        F.col("edge_type") == "track_artist",
    ).select("src_id", "dst_id")
    expected_artist: DataFrame = metadata.select(
        F.col("track_id").alias("src_id"),
        F.col("artist_id").alias("dst_id"),
    )
    _require_same_pairs(
        observed_artist, expected_artist, "track_artist mapping mismatch"
    )

    observed_release: DataFrame = graph_edges.where(
        F.col("edge_type") == "track_release",
    ).select("src_id", "dst_id")
    expected_release: DataFrame = metadata.where(
        F.col("release_7digitalid").isNotNull() & (F.col("release_7digitalid") > 0),
    ).select(
        F.col("track_id").alias("src_id"),
        F.col("release_7digitalid").cast("string").alias("dst_id"),
    )
    _require_same_pairs(
        observed_release, expected_release, "track_release mapping mismatch"
    )

    print(f"graph_edges total={sum(counts.values())}, counts={counts}")
    return {"graph_edges": sum(counts.values()), **counts}


def validate_manifest_lineage(
    prepared_dir: Path,
    tables: dict[str, DataFrame],
) -> dict[str, Any]:
    manifest: dict[str, Any] = read_manifest(prepared_dir / MANIFEST_NAME)
    require(
        manifest["artifact_type"] == ARTIFACT_TYPE, "manifest artifact type mismatch"
    )
    require(
        manifest["artifact_version"] == ARTIFACT_VERSION,
        "manifest artifact version mismatch",
    )
    require(manifest["status"] in {"initialized", "valid"}, "manifest status mismatch")

    config: dict[str, Any] = manifest["config"]
    require(
        config.get("shared_audio_contract_version") == SHARED_AUDIO_CONTRACT_VERSION,
        "manifest shared audio contract version mismatch",
    )
    require(
        config.get("shared_audio_feature_count") == SHARED_AUDIO_FEATURE_COUNT,
        "manifest shared audio feature count mismatch",
    )
    require(
        config.get("merlin_audio_feature_count") == MERLIN_AUDIO_FEATURE_COUNT,
        "manifest MERLIN audio projection count mismatch",
    )
    require(
        config.get("merlin_raw_feature_count") == MERLIN_RAW_FEATURE_COUNT,
        "manifest MERLIN raw feature count mismatch",
    )

    inputs: dict[str, dict[str, Any]] = {
        item["name"]: item for item in manifest["inputs"]
    }
    require(
        "audio_feature_contract" in inputs,
        "manifest missing audio feature contract lineage",
    )
    contract_input: dict[str, Any] = inputs["audio_feature_contract"]
    contract_path = Path(contract_input.get("path", ""))
    require(
        contract_path.name == FEATURE_CONTRACT_NAME and contract_path.is_file(),
        "manifest audio feature contract path is invalid",
    )
    require(
        contract_input.get("sha256") == sha256_file(contract_path),
        "manifest audio feature contract hash mismatch",
    )
    require(
        contract_input.get("contract_version") == SHARED_AUDIO_CONTRACT_VERSION,
        "manifest audio feature contract lineage version mismatch",
    )

    outputs: dict[str, dict[str, Any]] = {
        item["name"]: item for item in manifest["outputs"]
    }
    for name, frame in tables.items():
        require(name in outputs, f"manifest missing output {name}")
        require(
            outputs[name].get("schema_hash") == _schema_hash(frame),
            f"manifest schema hash mismatch for {name}",
        )
    return manifest


def mark_manifest_valid(
    prepared_dir: Path,
    manifest: dict[str, Any],
    statistics: dict[str, int],
) -> None:
    manifest["status"] = "valid"
    manifest["statistics"] = {
        **manifest["statistics"],
        "output_row_counts": statistics,
    }
    manifest["validation"] = {
        "passed": True,
        "validated_at_utc": utc_now(),
        "checks": [
            "layout",
            "schema",
            "track_identity",
            "finite_audio",
            "canonical_edge_types",
            "edge_counts",
            "edge_semantics",
            "manifest_lineage",
        ],
    }
    write_manifest(manifest, prepared_dir / MANIFEST_NAME, allow_replace=True)


def run_validation(
    spark: SparkSession,
    prepared_dir: Path,
    expected: ExpectedCounts,
    *,
    update_manifest: bool = True,
) -> dict[str, int]:
    validate_output_layout(prepared_dir)
    tables: dict[str, DataFrame] = read_outputs(spark, prepared_dir)
    validate_schema_contract(tables)
    manifest: dict[str, Any] = validate_manifest_lineage(prepared_dir, tables)
    statistics: dict[str, int] = {
        **validate_song_tables(tables, expected),
        **validate_graph_edges(tables, expected),
    }
    if update_manifest:
        mark_manifest_valid(prepared_dir, manifest, statistics)
    return statistics


def main() -> None:
    args: argparse.Namespace = parse_args()
    expected = ExpectedCounts(
        songs=args.expected_songs,
        track_release=args.expected_track_release,
        artist_term=args.expected_artist_term,
        artist_similarity=args.expected_artist_similarity,
    )
    spark: SparkSession = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        statistics: dict[str, int] = run_validation(spark, args.prepared, expected)
        print(f"MERLIN prepared data validation passed: {statistics}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
