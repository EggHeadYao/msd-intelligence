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


def validate_output_layout(prepared_dir: Path) -> None:
    actual: set[str] = {
        path.name for path in prepared_dir.iterdir() if path.is_dir()
    }
    missing: set[str] = set(EXPECTED_OUTPUT_DIRS) - actual
    extra: set[str] = actual - set(EXPECTED_OUTPUT_DIRS)
    require(not missing, f"prepared directory missing outputs: {sorted(missing)}")
    require(not extra, f"prepared directory has unexpected outputs: {sorted(extra)}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_columns(df: DataFrame, expected: tuple[str, ...], table_name: str) -> None:
    actual: set[str] = set(df.columns)
    expected_set: set[str] = set(expected)
    missing: set[str] = expected_set - actual
    extra: set[str] = actual - expected_set
    require(not missing, f"{table_name} missing columns: {sorted(missing)}")
    require(not extra, f"{table_name} has unexpected columns: {sorted(extra)}")


def require_type(
    df: DataFrame,
    table_name: str,
    column: str,
    allowed: tuple[type[T.DataType], ...],
) -> None:
    data_type: T.DataType = df.schema[column].dataType
    require(
        isinstance(data_type, allowed),
        f"{table_name}.{column} has type {data_type}, expected {allowed}",
    )


def count_distinct(df: DataFrame, col: str) -> int:
    return df.select(col).distinct().count()


def validate_schema_contract(tables: dict[str, DataFrame]) -> None:
    metadata: DataFrame = tables["songs_metadata"]
    audio: DataFrame = tables["song_audio_features"]
    song_terms: DataFrame = tables["song_terms"]
    graph_edges: DataFrame = tables["graph_edges"]

    require_columns(metadata, EXPECTED_METADATA_COLUMNS, "songs_metadata")
    require_columns(audio, EXPECTED_AUDIO_COLUMNS, "song_audio_features_raw")
    require_columns(song_terms, EXPECTED_SONG_TERMS_COLUMNS, "song_terms")
    require_columns(graph_edges, EXPECTED_GRAPH_EDGE_COLUMNS, "graph_edges")

    for col in (
        "track_id",
        "song_id",
        "title",
        "artist_id",
        "artist_name",
        "artist_mbid",
        "release",
        "album_key",
    ):
        require_type(metadata, "songs_metadata", col, (T.StringType,))
    for col in ("duration", "song_hotttnesss", "artist_hotttnesss", "artist_familiarity"):
        require_type(metadata, "songs_metadata", col, (T.NumericType,))
    for col in ("year", "has_year"):
        require_type(metadata, "songs_metadata", col, (T.NumericType,))

    require_type(audio, "song_audio_features_raw", "track_id", (T.StringType,))
    for col in EXPECTED_AUDIO_COLUMNS:
        if col != "track_id":
            require_type(audio, "song_audio_features_raw", col, (T.NumericType,))

    for col in EXPECTED_SONG_TERMS_COLUMNS:
        require_type(song_terms, "song_terms", col, (T.StringType,))

    for col in ("src_type", "src_id", "dst_type", "dst_id", "edge_type"):
        require_type(graph_edges, "graph_edges", col, (T.StringType,))
    require_type(graph_edges, "graph_edges", "weight", (T.NumericType,))
    require_type(graph_edges, "graph_edges", "directed", (T.BooleanType,))

    print(
        "schema_contract "
        f"metadata_cols={len(metadata.columns)}, audio_cols={len(audio.columns)}, "
        f"segment_cols={len(SEGMENT_FEATURE_COLUMNS)}, graph_cols={len(graph_edges.columns)}",
    )


def validate_song_tables(tables: dict[str, DataFrame]) -> None:
    metadata: DataFrame = tables["songs_metadata"]
    audio: DataFrame = tables["song_audio_features"]

    metadata_count: int = metadata.count()
    metadata_distinct: int = count_distinct(metadata, "track_id")
    audio_count: int = audio.count()
    audio_distinct: int = count_distinct(audio, "track_id")

    print(f"songs_metadata rows={metadata_count}, distinct_track_id={metadata_distinct}")
    print(f"song_audio_features rows={audio_count}, distinct_track_id={audio_distinct}")

    require(metadata_count == EXPECTED_SONGS, "songs_metadata row count mismatch")
    require(metadata_distinct == EXPECTED_SONGS, "songs_metadata track_id mismatch")
    require(audio_count == EXPECTED_SONGS, "song_audio_features row count mismatch")
    require(audio_distinct == EXPECTED_SONGS, "song_audio_features track_id mismatch")

    missing_audio: int = metadata.select("track_id").join(
        audio.select("track_id"),
        "track_id",
        "left_anti",
    ).count()
    missing_metadata: int = audio.select("track_id").join(
        metadata.select("track_id"),
        "track_id",
        "left_anti",
    ).count()
    print(f"metadata_minus_audio={missing_audio}, audio_minus_metadata={missing_metadata}")
    require(missing_audio == 0, "metadata contains track_id missing from audio")
    require(missing_metadata == 0, "audio contains track_id missing from metadata")

    bad_has_year: int = metadata.where(
        F.col("has_year").isNull() | ~F.col("has_year").isin(0, 1),
    ).count()
    bad_has_segments: int = tables["song_audio_features"].where(
        F.col("has_segments").isNull() | ~F.col("has_segments").isin(0, 1),
    ).count()
    require(bad_has_year == 0, "songs_metadata has_year contains values outside {0,1}")
    require(
        bad_has_segments == 0,
        "song_audio_features has_segments contains values outside {0,1}",
    )


def validate_terms(tables: dict[str, DataFrame]) -> None:
    song_terms: DataFrame = tables["song_terms"]
    rows: int = song_terms.count()
    distinct_tracks: int = count_distinct(song_terms, "track_id")
    distinct_terms: int = count_distinct(song_terms, "term")
    null_rows: int = song_terms.where(
        F.col("track_id").isNull()
        | F.col("artist_id").isNull()
        | F.col("term").isNull(),
    ).count()

    print(
        "song_terms "
        f"rows={rows}, distinct_track_id={distinct_tracks}, "
        f"distinct_terms={distinct_terms}, null_rows={null_rows}",
    )
    require(rows > 0, "song_terms is empty")
    require(distinct_tracks > 0, "song_terms has no tracks")
    require(distinct_terms > 0, "song_terms has no terms")
    require(null_rows == 0, "song_terms contains null key fields")


def validate_graph_edges(tables: dict[str, DataFrame]) -> None:
    graph_edges: DataFrame = tables["graph_edges"]
    counts: dict[str, int] = {
        row["edge_type"]: row["count"]
        for row in graph_edges.groupBy("edge_type").count().collect()
    }
    print(f"graph_edges rows={sum(counts.values())}, edge_type_counts={counts}")

    missing: set[str] = set(REQUIRED_EDGE_TYPES) - set(counts)
    extra: set[str] = set(counts) - set(REQUIRED_EDGE_TYPES)
    require(not missing, f"graph_edges missing edge types: {sorted(missing)}")
    require(not extra, f"graph_edges has unexpected edge types: {sorted(extra)}")
    for edge_type in REQUIRED_EDGE_TYPES:
        require(counts[edge_type] > 0, f"graph_edges {edge_type} is empty")

    require(counts["song_album"] == EXPECTED_SONGS, "song_album edge count mismatch")
    require(
        counts["song_artist"] == EXPECTED_SONGS,
        "song_artist edge count mismatch",
    )
    require(
        counts["artist_tag"] == EXPECTED_ARTIST_TAG_EDGES,
        "artist_tag edge count mismatch",
    )
    require(
        counts["song_year"] == EXPECTED_KNOWN_YEAR_SONGS,
        "song_year edge count mismatch",
    )
    require(
        counts["artist_similarity"] == EXPECTED_ARTIST_SIMILARITY_EDGES,
        "artist_similarity graph edge count mismatch",
    )
    artist_similarity: DataFrame = graph_edges.where(
        F.col("edge_type") == "artist_similarity",
    )
    distinct_artist_similarity_pairs: int = artist_similarity.select(
        "src_id",
        "dst_id",
    ).distinct().count()
    require(
        distinct_artist_similarity_pairs == EXPECTED_ARTIST_SIMILARITY_EDGES,
        "artist_similarity graph edges contain duplicates",
    )

    bad_node_types: int = graph_edges.where(
        ~F.col("src_type").isin(*EXPECTED_NODE_TYPES)
        | ~F.col("dst_type").isin(*EXPECTED_NODE_TYPES),
    ).count()
    require(bad_node_types == 0, "graph_edges contains unexpected node types")

    for edge_type, weight in EXPECTED_EDGE_WEIGHTS.items():
        bad_weight: int = graph_edges.where(
            (F.col("edge_type") == edge_type) & (F.col("weight") != weight),
        ).count()
        require(bad_weight == 0, f"graph_edges {edge_type} has unexpected weights")

    bad_directed: int = graph_edges.where(
        ((F.col("edge_type") == "artist_similarity") & (F.col("directed") != F.lit(True)))
        | ((F.col("edge_type") != "artist_similarity") & (F.col("directed") != F.lit(False))),
    ).count()
    require(bad_directed == 0, "graph_edges directed flags do not match edge semantics")

    bad_year_edges: int = graph_edges.where(
        (F.col("edge_type") == "song_year") & (F.col("dst_id") == "0"),
    ).count()
    require(bad_year_edges == 0, "song_year contains year=0 edges")

    null_rows: int = graph_edges.where(
        F.col("src_type").isNull()
        | F.col("src_id").isNull()
        | F.col("dst_type").isNull()
        | F.col("dst_id").isNull()
        | F.col("edge_type").isNull()
        | F.col("weight").isNull()
        | F.col("directed").isNull(),
    ).count()
    print(f"graph_edges null_rows={null_rows}")
    require(null_rows == 0, "graph_edges contains null fields")


def main() -> None:
    args: argparse.Namespace = parse_args()
    spark: SparkSession = create_spark(args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        validate_output_layout(args.prepared)
        tables: dict[str, DataFrame] = read_outputs(spark, args.prepared)
        validate_schema_contract(tables)
        validate_song_tables(tables)
        validate_terms(tables)
        validate_graph_edges(tables)
        print("MERLIN prepared data validation passed.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
