from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MERLIN tables from raw MSD Parquet files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("parquets"),
        help="Raw Parquet directory (default: parquets)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parquets/prepared"),
        help="Prepared output directory (default: parquets/prepared)",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=32,
        help="Spark SQL shuffle partitions (default: 32)",
    )
    return parser.parse_args()


def create_spark(shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("MerlinPrepare")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def resolve_music_dir(input_dir: Path) -> Path:
    candidates: tuple[Path, ...] = (
        input_dir / "musics",
        input_dir / "_extracted" / "parquets" / "musics",
    )
    for candidate in candidates:
        if list(candidate.glob("features_*.parquet")):
            return candidate
    checked: str = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No musics/features_*.parquet found under: {checked}")


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def parquet_batch_paths(directory: Path, pattern: str) -> list[str]:
    matches: list[Path] = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {directory / pattern}")
    return [spark_path(path) for path in matches]


def read_inputs(spark: SparkSession, input_dir: Path) -> dict[str, DataFrame]:
    music_dir: Path = resolve_music_dir(input_dir)
    return {
        "songs_scalar": spark.read.parquet(
            spark_path(input_dir / "songs_scalar.parquet"),
        ),
        "track_metadata": spark.read.parquet(
            spark_path(input_dir / "track_metadata.parquet"),
        ),
        "artist_term": spark.read.parquet(spark_path(input_dir / "artist_term.parquet")),
        "artist_similarity_edges": spark.read.parquet(
            spark_path(input_dir / "artist_similarity_edges.parquet"),
        ),
        "features": spark.read.parquet(
            *parquet_batch_paths(music_dir, "features_*.parquet"),
        ),
        "terms": spark.read.parquet(
            *parquet_batch_paths(music_dir, "terms_*.parquet"),
        ),
    }


def build_songs_metadata(inputs: dict[str, DataFrame]) -> DataFrame:
    metadata: DataFrame = inputs["track_metadata"].alias("m")
    scalar: DataFrame = inputs["songs_scalar"].select(
        "track_id",
        "song_hotttnesss",
    ).alias("s")

    return (
        metadata.join(scalar, "track_id", "inner")
        .select(
            F.col("track_id"),
            F.col("m.song_id"),
            F.col("m.title"),
            F.col("m.artist_id"),
            F.col("m.artist_name"),
            F.col("m.artist_mbid"),
            F.col("m.release"),
            F.concat(
                F.col("m.artist_id"),
                F.lit("::"),
                F.coalesce(F.col("m.release"), F.lit("")),
            ).alias("album_key"),
            F.col("m.duration"),
            F.col("m.year").cast("int").alias("year"),
            (F.col("m.year") > 0).cast("int").alias("has_year"),
            F.col("s.song_hotttnesss"),
            F.col("m.artist_hotttnesss"),
            F.col("m.artist_familiarity"),
        )
    )


def build_song_audio_features(inputs: dict[str, DataFrame]) -> DataFrame:
    scalar_cols: list[str] = [
        "track_id",
        "danceability",
        "energy",
        "loudness",
        "tempo",
        "duration",
        "key",
        "mode",
        "time_signature",
    ]
    scalar: DataFrame = inputs["songs_scalar"].select(*scalar_cols)
    features: DataFrame = inputs["features"]

    feature_cols: list[str] = [c for c in features.columns if c != "track_id"]
    return (
        scalar.join(features, "track_id", "inner")
        .select(*scalar_cols, *[F.col(c) for c in feature_cols])
    )


def build_song_terms(
    songs_metadata: DataFrame,
    inputs: dict[str, DataFrame],
) -> DataFrame:
    terms: DataFrame = inputs["terms"].select("track_id", "term")
    songs: DataFrame = songs_metadata.select("track_id", "artist_id")
    return (
        terms.join(songs, "track_id", "inner")
        .select("track_id", "artist_id", "term")
    )

