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


def build_artist_similarity_edges(inputs: dict[str, DataFrame]) -> DataFrame:
    return (
        inputs["artist_similarity_edges"]
        .select(
            F.col("src").alias("src_artist_id"),
            F.col("dst").alias("dst_artist_id"),
        )
    )


def edge_frame(
    df: DataFrame,
    src_type: str,
    src_col: str,
    dst_type: str,
    dst_col: str,
    edge_type: str,
    weight: float,
    directed: bool,
) -> DataFrame:
    return df.select(
        F.lit(src_type).alias("src_type"),
        F.col(src_col).cast("string").alias("src_id"),
        F.lit(dst_type).alias("dst_type"),
        F.col(dst_col).cast("string").alias("dst_id"),
        F.lit(edge_type).alias("edge_type"),
        F.lit(weight).cast("double").alias("weight"),
        F.lit(directed).cast("boolean").alias("directed"),
    ).where(F.col("src_id").isNotNull() & F.col("dst_id").isNotNull())


def build_graph_edges(
    songs_metadata: DataFrame,
    song_terms: DataFrame,
    artist_similarity_edges: DataFrame,
    inputs: dict[str, DataFrame],
) -> DataFrame:
    song_artist: DataFrame = edge_frame(
        songs_metadata,
        "song",
        "track_id",
        "artist",
        "artist_id",
        "song_artist",
        1.0,
        False,
    )

    albums: DataFrame = songs_metadata.where(
        F.col("release").isNotNull() & (F.col("release") != ""),
    )
    song_album: DataFrame = edge_frame(
        albums,
        "song",
        "track_id",
        "album",
        "album_key",
        "song_album",
        0.8,
        False,
    )

    song_tag: DataFrame = edge_frame(
        song_terms,
        "song",
        "track_id",
        "tag",
        "term",
        "song_tag",
        0.6,
        False,
    )

    artist_terms: DataFrame = inputs["artist_term"].select("artist_id", "term")
    artist_tag: DataFrame = edge_frame(
        artist_terms,
        "artist",
        "artist_id",
        "tag",
        "term",
        "artist_tag",
        0.6,
        False,
    )


    return (
        song_artist.unionByName(song_album)
        .unionByName(song_tag)
        .unionByName(artist_tag)
        .unionByName(song_year)
        .unionByName(artist_similarity)
    )

