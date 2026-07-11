from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


EXPECTED_SONGS: int = 1_000_000
EXPECTED_ARTIST_SIMILARITY_EDGES: int = 2_201_916
EXPECTED_ARTIST_TAG_EDGES: int = 1_109_381
EXPECTED_KNOWN_YEAR_SONGS: int = 515_576
EXPECTED_OUTPUT_DIRS: frozenset[str] = frozenset(
    {
        "songs_metadata.parquet",
        "song_audio_features_raw.parquet",
        "song_terms.parquet",
        "graph_edges.parquet",
    },
)
REQUIRED_EDGE_TYPES: frozenset[str] = frozenset(
    {
        "song_artist",
        "song_album",
        "song_tag",
        "artist_tag",
        "song_year",
        "artist_similarity",
    },
)
SEGMENT_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [
        f"{prefix}_{stat}_{i}"
        for prefix in ("pitch", "timbre")
        for stat in ("mean", "std", "min", "max")
        for i in range(12)
    ]
    + [f"loudness_{stat}" for stat in ("mean", "std", "min", "max")]
)
AUDIO_SCALAR_COLUMNS: tuple[str, ...] = (
    "track_id",
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "duration",
    "key",
    "mode",
    "time_signature",
)
EXPECTED_AUDIO_COLUMNS: tuple[str, ...] = (
    *AUDIO_SCALAR_COLUMNS,
    *SEGMENT_FEATURE_COLUMNS,
    "has_segments",
)
EXPECTED_METADATA_COLUMNS: tuple[str, ...] = (
    "track_id",
    "song_id",
    "title",
    "artist_id",
    "artist_name",
    "artist_mbid",
    "release",
    "album_key",
    "duration",
    "year",
    "has_year",
    "song_hotttnesss",
    "artist_hotttnesss",
    "artist_familiarity",
)
EXPECTED_SONG_TERMS_COLUMNS: tuple[str, ...] = ("track_id", "artist_id", "term")
EXPECTED_GRAPH_EDGE_COLUMNS: tuple[str, ...] = (
    "src_type",
    "src_id",
    "dst_type",
    "dst_id",
    "weight",
    "directed",
    "edge_type",
)
EXPECTED_NODE_TYPES: frozenset[str] = frozenset({"song", "artist", "album", "tag", "year"})
EXPECTED_EDGE_WEIGHTS: dict[str, float] = {
    "song_artist": 1.0,
    "song_album": 0.8,
    "song_tag": 0.6,
    "artist_tag": 0.6,
    "song_year": 0.4,
    "artist_similarity": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate prepared MERLIN Parquet tables.",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("parquets/prepared"),
        help="Prepared MERLIN directory (default: parquets/prepared)",
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
        SparkSession.builder.appName("MerlinValidate")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def read_outputs(spark: SparkSession, prepared_dir: Path) -> dict[str, DataFrame]:
    return {
        "songs_metadata": spark.read.parquet(
            spark_path(prepared_dir / "songs_metadata.parquet"),
        ),
        "song_audio_features": spark.read.parquet(
            spark_path(prepared_dir / "song_audio_features_raw.parquet"),
        ),
        "song_terms": spark.read.parquet(spark_path(prepared_dir / "song_terms.parquet")),
        "graph_edges": spark.read.parquet(spark_path(prepared_dir / "graph_edges.parquet")),
    }

