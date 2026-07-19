# ruff: noqa: T201
"""Build the canonical MERLIN metadata, audio, and graph tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from merlin.artifacts.contract import (
    initialize_output_dir,
    make_manifest,
    sha256_file,
    write_manifest,
)
from merlin.prepare.contract import (
    ARTIFACT_TYPE,
    ARTIFACT_VERSION,
    ARTIST_SIMILARITY_COLUMNS,
    ARTIST_TERM_COLUMNS,
    AUDIO_COLUMNS,
    EDGE_TYPES,
    EXTRACTED_AUDIO_COLUMNS,
    MANIFEST_NAME,
    SUMMARY_AUDIO_COLUMNS,
    SUMMARY_COLUMNS,
    TRACK_METADATA_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical MERLIN prepared tables.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../parquets_new"),
        help="Directory containing all prepared inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../parquets_new/prepared"),
        help="New prepared output directory.",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=32,
        help="Spark SQL shuffle partitions.",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Reset an existing matching MERLIN-owned output directory.",
    )
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinPrepare")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def parquet_batch_paths(directory: Path, pattern: str) -> list[str]:
    matches: list[Path] = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {directory / pattern}")
    return [spark_path(path) for path in matches]


def resolve_input_paths(input_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "songs_scalar": input_dir / "songs_scalar.parquet",
        "features": input_dir / "musics",
        "track_metadata": input_dir / "track_metadata.parquet",
        "artist_term": input_dir / "artist_term.parquet",
        "artist_similarity": input_dir / "artist_similarity_edges.parquet",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} input: {path}")
    return paths


def read_inputs(spark: SparkSession, paths: dict[str, Path]) -> dict[str, DataFrame]:
    return {
        "songs_scalar": spark.read.parquet(spark_path(paths["songs_scalar"])),
        "features": spark.read.parquet(
            *parquet_batch_paths(paths["features"], "features_*.parquet"),
        ),
        "track_metadata": spark.read.parquet(spark_path(paths["track_metadata"])),
        "artist_term": spark.read.parquet(spark_path(paths["artist_term"])),
        "artist_similarity": spark.read.parquet(
            spark_path(paths["artist_similarity"]),
        ),
    }


def require_exact_columns(
    frame: DataFrame,
    expected: tuple[str, ...],
    table_name: str,
) -> None:
    actual: tuple[str, ...] = tuple(frame.columns)
    if actual == expected:
        return
    missing: list[str] = sorted(set(expected) - set(actual))
    extra: list[str] = sorted(set(actual) - set(expected))
    raise ValueError(
        f"{table_name} schema mismatch: missing={missing}, extra={extra}, "
        f"ordered_columns_match={set(actual) == set(expected)}",
    )


def validate_input_schemas(inputs: dict[str, DataFrame]) -> None:
    require_exact_columns(inputs["songs_scalar"], SUMMARY_COLUMNS, "songs_scalar")
    require_exact_columns(inputs["features"], EXTRACTED_AUDIO_COLUMNS, "features")
    require_exact_columns(
        inputs["track_metadata"],
        TRACK_METADATA_COLUMNS,
        "track_metadata",
    )
    require_exact_columns(inputs["artist_term"], ARTIST_TERM_COLUMNS, "artist_term")
    require_exact_columns(
        inputs["artist_similarity"],
        ARTIST_SIMILARITY_COLUMNS,
        "artist_similarity_edges",
    )


def _validate_unique_keys(
    frame: DataFrame,
    columns: tuple[str, ...],
    table_name: str,
) -> int:
    invalid = F.lit(False)
    for column in columns:
        invalid = invalid | F.col(column).isNull() | (F.col(column).cast("string") == "")
    if frame.where(invalid).limit(1).count():
        raise ValueError(f"{table_name} contains null or empty key fields")

    rows: int = frame.count()
    distinct_rows: int = frame.select(*columns).distinct().count()
    if rows != distinct_rows:
        raise ValueError(
            f"{table_name} key is not unique: rows={rows}, distinct={distinct_rows}",
        )
    return rows


def validate_input_data(inputs: dict[str, DataFrame]) -> dict[str, int]:
    counts: dict[str, int] = {
        "songs_scalar": _validate_unique_keys(
            inputs["songs_scalar"],
            ("track_id",),
            "songs_scalar",
        ),
        "features": _validate_unique_keys(
            inputs["features"],
            ("track_id",),
            "features",
        ),
        "track_metadata": _validate_unique_keys(
            inputs["track_metadata"],
            ("track_id",),
            "track_metadata",
        ),
        "artist_term": _validate_unique_keys(
            inputs["artist_term"],
            ("artist_id", "term"),
            "artist_term",
        ),
        "artist_similarity": _validate_unique_keys(
            inputs["artist_similarity"],
            ("src", "dst"),
            "artist_similarity_edges",
        ),
    }

    feature_ids: DataFrame = inputs["features"].select("track_id")
    for table_name in ("songs_scalar", "track_metadata"):
        missing: int = feature_ids.join(
            inputs[table_name].select("track_id"),
            "track_id",
            "left_anti",
        ).limit(1).count()
        if missing:
            raise ValueError(f"features contains track_id missing from {table_name}")

    identity_mismatch: int = (
        feature_ids.join(
            inputs["songs_scalar"].select("track_id", "song_id", "artist_id"),
            "track_id",
        )
        .join(
            inputs["track_metadata"].select(
                "track_id",
                F.col("song_id").alias("metadata_song_id"),
                F.col("artist_id").alias("metadata_artist_id"),
            ),
            "track_id",
        )
        .where(
            (F.col("song_id") != F.col("metadata_song_id"))
            | (F.col("artist_id") != F.col("metadata_artist_id")),
        )
        .limit(1)
        .count()
    )
    if identity_mismatch:
        raise ValueError("songs_scalar and track_metadata disagree on song/artist IDs")
    return counts


def build_songs_metadata(inputs: dict[str, DataFrame]) -> DataFrame:
    catalog: DataFrame = inputs["features"].select("track_id")
    summary: DataFrame = inputs["songs_scalar"].alias("s")
    metadata: DataFrame = inputs["track_metadata"].select(
        "track_id",
        "artist_mbid",
    ).alias("m")

    return (
        catalog.join(summary, "track_id", "inner")
        .join(metadata, "track_id", "inner")
        .select(
            F.col("track_id"),
            F.col("s.song_id").alias("song_id"),
            F.col("s.title").alias("title"),
            F.col("s.artist_id").alias("artist_id"),
            F.col("s.artist_name").alias("artist_name"),
            F.col("m.artist_mbid").alias("artist_mbid"),
            F.col("s.release").alias("release"),
            F.col("s.release_7digitalid").cast("long").alias("release_7digitalid"),
            F.col("s.track_7digitalid").cast("long").alias("track_7digitalid"),
            F.col("s.duration").cast("double").alias("duration"),
            F.col("s.year").cast("int").alias("year"),
            F.coalesce((F.col("s.year") > 0).cast("int"), F.lit(0)).alias(
                "has_year",
            ),
            F.col("s.song_hotttnesss").cast("double").alias("song_hotttnesss"),
            F.col("s.artist_hotttnesss").cast("double").alias("artist_hotttnesss"),
            F.col("s.artist_familiarity").cast("double").alias("artist_familiarity"),
        )
    )


def build_song_audio_features(inputs: dict[str, DataFrame]) -> DataFrame:
    summary: DataFrame = inputs["songs_scalar"].select(*SUMMARY_AUDIO_COLUMNS)
    features: DataFrame = inputs["features"]
    return summary.join(features, "track_id", "inner").select(*AUDIO_COLUMNS)


def edge_frame(
    frame: DataFrame,
    src_type: str,
    src_column: str,
    dst_type: str,
    dst_column: str,
    edge_type: str,
    directed: bool,
) -> DataFrame:
    return frame.select(
        F.lit(src_type).alias("src_type"),
        F.col(src_column).cast("string").alias("src_id"),
        F.lit(dst_type).alias("dst_type"),
        F.col(dst_column).cast("string").alias("dst_id"),
        F.lit(directed).cast("boolean").alias("directed"),
        F.lit(edge_type).alias("edge_type"),
    )


def build_graph_edge_frames(
    songs_metadata: DataFrame,
    inputs: dict[str, DataFrame],
) -> dict[str, DataFrame]:
    track_release_source: DataFrame = songs_metadata.where(
        F.col("release_7digitalid").isNotNull()
        & (F.col("release_7digitalid") > 0),
    )
    frames: dict[str, DataFrame] = {
        "track_artist": edge_frame(
            songs_metadata,
            "track",
            "track_id",
            "artist",
            "artist_id",
            "track_artist",
            False,
        ),
        "track_release": edge_frame(
            track_release_source,
            "track",
            "track_id",
            "release",
            "release_7digitalid",
            "track_release",
            False,
        ),
        "artist_term": edge_frame(
            inputs["artist_term"],
            "artist",
            "artist_id",
            "term",
            "term",
            "artist_term",
            False,
        ),
        "artist_similarity": edge_frame(
            inputs["artist_similarity"],
            "artist",
            "src",
            "artist",
            "dst",
            "artist_similarity",
            True,
        ),
    }
    if tuple(frames) != EDGE_TYPES:
        raise RuntimeError("Graph edge order does not match the prepared contract")
    return frames


def write_table(frame: DataFrame, path: Path) -> None:
    frame.write.mode("errorifexists").parquet(spark_path(path))


def write_graph_edges(edge_frames: dict[str, DataFrame], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    root_uri: str = spark_path(path)
    for edge_type, frame in edge_frames.items():
        frame.drop("edge_type").write.mode("errorifexists").parquet(
            f"{root_uri}/edge_type={edge_type}",
        )


def _schema_descriptor(frame: DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": field.dataType.simpleString(),
        }
        for field in frame.schema.fields
    ]


def _schema_hash(frame: DataFrame) -> str:
    encoded: bytes = json.dumps(
        _schema_descriptor(frame),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_code_state() -> dict[str, Any]:
    repository: Path = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        )
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        dirty = True
    return {"commit": commit, "dirty": dirty}


def write_initialized_manifest(
    output_dir: Path,
    paths: dict[str, Path],
    inputs: dict[str, DataFrame],
    outputs: dict[str, DataFrame],
    input_counts: dict[str, int],
    shuffle_partitions: int,
) -> None:
    defaults_path: Path = Path(__file__).resolve().parents[1] / "artifacts" / "merlin_v3_defaults.json"
    manifest = make_manifest(
        artifact_type=ARTIFACT_TYPE,
        artifact_version=ARTIFACT_VERSION,
        status="initialized",
        code=_git_code_state(),
        config={
            "defaults_path": str(defaults_path),
            "defaults_sha256": sha256_file(defaults_path),
            "shuffle_partitions": shuffle_partitions,
            "graph_is_weighted": False,
            "edge_types": list(EDGE_TYPES),
        },
        inputs=[
            {
                "name": name,
                "path": str(paths[name].resolve()),
                "row_count": input_counts[name],
                "schema_hash": _schema_hash(inputs[name]),
            }
            for name in paths
        ],
        outputs=[
            {
                "name": name,
                "path": str((output_dir / path_name).resolve()),
                "schema_hash": _schema_hash(outputs[name]),
                "columns": list(outputs[name].columns),
            }
            for name, path_name in (
                ("songs_metadata", "songs_metadata.parquet"),
                ("song_audio_features", "song_audio_features_raw.parquet"),
                ("graph_edges", "graph_edges.parquet"),
            )
        ],
        statistics={"input_row_counts": input_counts},
        validation={"passed": False, "checks": []},
    )
    write_manifest(manifest, output_dir / MANIFEST_NAME)


def run_prepare(
    spark: SparkSession,
    input_dir: Path,
    output_dir: Path,
    *,
    shuffle_partitions: int = 32,
    reset_output: bool = False,
) -> Path:
    paths: dict[str, Path] = resolve_input_paths(input_dir)
    inputs: dict[str, DataFrame] = read_inputs(spark, paths)
    validate_input_schemas(inputs)
    input_counts: dict[str, int] = validate_input_data(inputs)

    songs_metadata: DataFrame = build_songs_metadata(inputs)
    song_audio_features: DataFrame = build_song_audio_features(inputs)
    graph_edge_frames: dict[str, DataFrame] = build_graph_edge_frames(
        songs_metadata,
        inputs,
    )

    prepared_root: Path = initialize_output_dir(
        output_dir,
        artifact_type=ARTIFACT_TYPE,
        artifact_version=ARTIFACT_VERSION,
        input_paths=list(paths.values()),
        reset=reset_output,
    )
    write_table(songs_metadata, prepared_root / "songs_metadata.parquet")
    write_table(
        song_audio_features,
        prepared_root / "song_audio_features_raw.parquet",
    )
    write_graph_edges(graph_edge_frames, prepared_root / "graph_edges.parquet")

    graph_edges: DataFrame = graph_edge_frames[EDGE_TYPES[0]]
    for edge_type in EDGE_TYPES[1:]:
        graph_edges = graph_edges.unionByName(graph_edge_frames[edge_type])
    outputs: dict[str, DataFrame] = {
        "songs_metadata": songs_metadata,
        "song_audio_features": song_audio_features,
        "graph_edges": graph_edges,
    }
    write_initialized_manifest(
        prepared_root,
        paths,
        inputs,
        outputs,
        input_counts,
        shuffle_partitions,
    )
    return prepared_root


def main() -> None:
    args: argparse.Namespace = parse_args()
    spark: SparkSession = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        output: Path = run_prepare(
            spark,
            args.input,
            args.output,
            shuffle_partitions=args.shuffle_partitions,
            reset_output=args.reset_output,
        )
        print(f"Prepared tables written to {output}")
        print("Run merlin.prepare.validate before using this artifact downstream.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
