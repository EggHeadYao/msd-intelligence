"""Prepare graph inputs and labels for masked-artist C2 retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from functools import reduce
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


EXPERIMENT_VERSION = "c2_l1_2_masked_relation_v1"
STRATA = ("2", "3_5", "6_20", "21_plus")
QUERY_COLUMNS = (
    "query_track_id",
    "artist_id",
    "song_id",
    "release_7digitalid",
    "song_hotttnesss",
    "artist_track_count",
    "positive_count",
    "release_degree",
    "candidate_catalog_size",
    "connectable",
    "stratum",
)
GRAPH_COLUMNS = (
    "src_id",
    "dst_id",
    "src_type",
    "dst_type",
    "edge_type",
    "directed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--output-partitions", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def create_spark(shuffle_partitions: int) -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    return (
        SparkSession.builder.appName("MerlinC2MaskedArtistPrepare")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.pyspark.python", sys.executable)
        .getOrCreate()
    )


def allocate_balanced_quotas(
    available: dict[str, int],
    requested: int,
) -> dict[str, int]:
    """Allocate near-equal deterministic quotas, redistributing short strata."""
    if requested <= 0:
        raise ValueError("requested query count must be positive")
    if set(available) != set(STRATA):
        raise ValueError(f"availability must contain exactly {STRATA}")
    if any(value < 0 for value in available.values()):
        raise ValueError("stratum availability cannot be negative")
    if sum(available.values()) < requested:
        raise ValueError("not enough eligible artists for the requested queries")

    quotas = {name: 0 for name in STRATA}
    remaining = requested
    while remaining:
        progressed = False
        for name in STRATA:
            if quotas[name] >= available[name]:
                continue
            quotas[name] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("query quota allocation made no progress")
    return quotas


def _require_columns(frame: DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = set(required) - set(frame.columns)
    require(not missing, f"{name} is missing columns: {sorted(missing)}")


def _valid_string(column: str) -> F.Column:
    return F.col(column).isNotNull() & (F.length(F.trim(F.col(column))) > 0)


def enrich_metadata(metadata: DataFrame) -> tuple[DataFrame, int]:
    required = (
        "track_id",
        "artist_id",
        "song_id",
        "release_7digitalid",
        "song_hotttnesss",
    )
    _require_columns(metadata, required, "songs metadata")
    metadata = metadata.select(*required)

    counts = metadata.agg(
        F.count("*").alias("rows"),
        F.countDistinct("track_id").alias("tracks"),
    ).first()
    require(counts is not None, "songs metadata is empty")
    total_tracks = int(counts["rows"])
    require(total_tracks == int(counts["tracks"]), "track_id is not unique")
    require(
        metadata.where(~_valid_string("track_id")).limit(1).count() == 0,
        "track_id contains null or empty values",
    )

    base = metadata.withColumn(
        "song_key",
        F.when(
            _valid_string("song_id"),
            F.concat(F.lit("song:"), F.col("song_id")),
        ).otherwise(F.concat(F.lit("track:"), F.col("track_id"))),
    )
    artist_stats = (
        base.where(_valid_string("artist_id"))
        .groupBy("artist_id")
        .agg(
            F.count("*").cast("int").alias("artist_track_count"),
            F.countDistinct("song_key").cast("int").alias("artist_song_count"),
        )
    )
    artist_song_stats = (
        base.where(_valid_string("artist_id"))
        .groupBy("artist_id", "song_key")
        .agg(F.count("*").cast("int").alias("artist_same_song_count"))
    )
    catalog_song_stats = base.groupBy("song_key").agg(
        F.count("*").cast("int").alias("catalog_same_song_count"),
    )
    release_stats = (
        base.where(
            F.col("release_7digitalid").isNotNull() & (F.col("release_7digitalid") > 0)
        )
        .groupBy("release_7digitalid")
        .agg(F.countDistinct("track_id").cast("int").alias("release_degree"))
    )

    enriched = (
        base.join(artist_stats, "artist_id", "inner")
        .join(artist_song_stats, ["artist_id", "song_key"], "inner")
        .join(catalog_song_stats, "song_key", "inner")
        .join(release_stats, "release_7digitalid", "left")
        .fillna({"release_degree": 0})
        .withColumn(
            "positive_count",
            F.col("artist_track_count") - F.col("artist_same_song_count"),
        )
        .withColumn(
            "candidate_catalog_size",
            F.lit(total_tracks) - F.col("catalog_same_song_count"),
        )
        .withColumn(
            "connectable",
            (F.col("release_degree") > 1).cast("boolean"),
        )
        .withColumn(
            "stratum",
            F.when(F.col("artist_track_count") == 2, F.lit("2"))
            .when(F.col("artist_track_count") <= 5, F.lit("3_5"))
            .when(F.col("artist_track_count") <= 20, F.lit("6_20"))
            .otherwise(F.lit("21_plus")),
        )
        .where(F.col("positive_count") > 0)
    )
    return enriched, total_tracks


def select_queries(
    enriched: DataFrame,
    requested: int,
    seed: int,
) -> tuple[DataFrame, dict[str, int], dict[str, int]]:
    track_order = Window.partitionBy("artist_id").orderBy(
        F.xxhash64(F.lit(seed), "artist_id", "track_id"),
        F.col("track_id"),
    )
    one_per_artist = (
        enriched.withColumn("_track_rank", F.row_number().over(track_order))
        .where(F.col("_track_rank") == 1)
        .drop("_track_rank")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    available_rows = one_per_artist.groupBy("stratum").count().collect()
    available = {name: 0 for name in STRATA}
    for row in available_rows:
        available[str(row["stratum"])] = int(row["count"])
    quotas = allocate_balanced_quotas(available, requested)

    selected_parts: list[DataFrame] = []
    for offset, name in enumerate(STRATA):
        selected_parts.append(
            one_per_artist.where(F.col("stratum") == name)
            .orderBy(
                F.xxhash64(F.lit(seed + 1 + offset), "artist_id"),
                F.col("artist_id"),
            )
            .limit(quotas[name])
        )
    queries = (
        reduce(DataFrame.unionByName, selected_parts)
        .withColumnRenamed("track_id", "query_track_id")
        .select(
            "query_track_id",
            "artist_id",
            "song_id",
            "song_key",
            "release_7digitalid",
            "song_hotttnesss",
            "artist_track_count",
            "positive_count",
            "release_degree",
            "candidate_catalog_size",
            "connectable",
            "stratum",
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    require(queries.count() == requested, "query selection returned the wrong count")
    require(
        queries.select("artist_id").distinct().count() == requested,
        "query selection contains duplicate artists",
    )
    one_per_artist.unpersist()
    return queries, available, quotas


def build_positives(metadata: DataFrame, queries: DataFrame) -> DataFrame:
    candidates = metadata.select(
        "track_id",
        "artist_id",
        F.when(
            _valid_string("song_id"),
            F.concat(F.lit("song:"), F.col("song_id")),
        )
        .otherwise(F.concat(F.lit("track:"), F.col("track_id")))
        .alias("song_key"),
    ).alias("candidate")
    query_labels = queries.select(
        "query_track_id",
        "artist_id",
        F.col("song_key").alias("query_song_key"),
    ).alias("query")
    return (
        F.broadcast(query_labels)
        .join(
            candidates,
            F.col("query.artist_id") == F.col("candidate.artist_id"),
            "inner",
        )
        .where(
            (F.col("candidate.track_id") != F.col("query.query_track_id"))
            & (F.col("candidate.song_key") != F.col("query.query_song_key"))
        )
        .select(
            F.col("query.query_track_id").alias("query_track_id"),
            F.col("candidate.track_id").alias("positive_track_id"),
        )
    )


def mask_track_artist_edges(edges: DataFrame, queries: DataFrame) -> DataFrame:
    _require_columns(edges, GRAPH_COLUMNS, "graph edges")
    query_ids = F.broadcast(
        queries.select(F.col("query_track_id").alias("masked_track_id")),
    )
    artist_edges = edges.where(F.col("edge_type") == "track_artist")
    kept_artist_edges = artist_edges.join(
        query_ids,
        artist_edges.src_id == query_ids.masked_track_id,
        "left_anti",
    )
    return edges.where(F.col("edge_type") != "track_artist").unionByName(
        kept_artist_edges,
    )


def edge_counts(edges: DataFrame) -> dict[str, int]:
    return {
        str(row["edge_type"]): int(row["count"])
        for row in edges.groupBy("edge_type").count().collect()
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _prepare_output(output: Path, overwrite: bool) -> Path:
    output = output.resolve()
    staging = output.with_name(output.name + ".tmp")
    existing = [path for path in (output, staging) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"masked-artist output already exists: {existing}")
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    staging.mkdir(parents=True)
    return staging


def main() -> None:
    args = parse_args()
    require(args.queries > 0, "query count must be positive")
    require(args.shuffle_partitions > 0, "shuffle partitions must be positive")
    require(args.output_partitions > 0, "output partitions must be positive")
    require(args.metadata.is_dir(), "metadata path must be a Parquet directory")
    require(args.graph.is_dir(), "graph path must be a Parquet directory")

    staging = _prepare_output(args.output, args.overwrite)
    spark = create_spark(args.shuffle_partitions)
    try:
        metadata = spark.read.parquet(spark_path(args.metadata)).persist(
            StorageLevel.MEMORY_AND_DISK,
        )
        enriched, total_tracks = enrich_metadata(metadata)
        queries, available, quotas = select_queries(
            enriched,
            args.queries,
            args.seed,
        )
        positives = build_positives(metadata, queries).persist(
            StorageLevel.MEMORY_AND_DISK,
        )

        positive_rows = positives.count()
        expected_positive_rows = int(
            queries.agg(F.sum("positive_count").alias("count")).first()["count"],
        )
        require(
            positive_rows == expected_positive_rows,
            "positive pair count does not match the query manifest",
        )

        edges = spark.read.parquet(spark_path(args.graph)).persist(
            StorageLevel.MEMORY_AND_DISK,
        )
        before_counts = edge_counts(edges)
        masked_edges = mask_track_artist_edges(edges, queries).persist(
            StorageLevel.MEMORY_AND_DISK,
        )
        after_counts = edge_counts(masked_edges)
        require(
            before_counts.get("track_artist", 0) - after_counts.get("track_artist", 0)
            == args.queries,
            "masking did not remove exactly one track_artist edge per query",
        )
        for edge_type, count in before_counts.items():
            if edge_type != "track_artist":
                require(
                    after_counts.get(edge_type) == count,
                    f"masking changed unrelated edge count: {edge_type}",
                )

        track_nodes = (
            masked_edges.where(F.col("src_type") == "track")
            .select(F.col("src_id").alias("track_id"))
            .unionByName(
                masked_edges.where(F.col("dst_type") == "track").select(
                    F.col("dst_id").alias("track_id"),
                ),
            )
            .distinct()
            .count()
        )
        query_rows = queries.select(*QUERY_COLUMNS).orderBy("query_track_id")
        query_ids = [
            str(row["query_track_id"])
            for row in query_rows.select("query_track_id").collect()
        ]
        query_sha256 = hashlib.sha256(
            ("\n".join(query_ids) + "\n").encode("ascii"),
        ).hexdigest()
        connectable = queries.where(F.col("connectable")).count()

        prepared = staging / "prepared"
        prepared.mkdir()
        query_rows.coalesce(1).write.mode("error").parquet(
            spark_path(staging / "queries.parquet"),
        )
        positives.repartition(
            min(args.output_partitions, args.shuffle_partitions),
            "query_track_id",
        ).write.mode("error").parquet(spark_path(staging / "positives.parquet"))
        masked_edges.repartition(
            args.output_partitions,
            "edge_type",
        ).write.mode("error").parquet(
            spark_path(prepared / "graph_edges.parquet"),
        )
        shutil.copytree(
            args.metadata.resolve(),
            prepared / "songs_metadata.parquet",
        )
    finally:
        spark.stop()
