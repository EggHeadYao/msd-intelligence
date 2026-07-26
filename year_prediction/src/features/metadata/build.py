from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from contract import (  # noqa: E402
    AUDIT_COLUMNS,
    BASE_METADATA_COLUMNS,
    ERA_COLUMNS,
    GRAPH_COLUMNS,
    GRAPH_RANK_COLUMNS,
    GRAPH_TOP_K_COLUMNS,
    LOCATION_COLUMNS,
    SCALAR_COLUMNS,
    SCALAR_MISSING_COLUMNS,
    SIMILARITY_TOP_K,
    TAG_COUNT_COLUMNS,
    TAG_PRIOR_COLUMNS,
    indicator_columns,
    order_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata feature views")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("parquets/year_prediction/raw/metadata"),
    )
    parser.add_argument(
        "--scalar",
        type=Path,
        default=Path("parquets/year_prediction/raw/songs_scalar.parquet"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("parquets/year_prediction/dataset/labelled_tracks.parquet"),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("parquets/year_prediction/features/full_tabular.parquet"),
    )
    parser.add_argument(
        "--audio-manifest",
        type=Path,
        default=Path("parquets/year_prediction/features/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parquets/year_prediction/features/metadata"),
    )
    parser.add_argument("--top-terms", type=int, default=256)
    parser.add_argument("--top-mbtags", type=int, default=64)
    parser.add_argument("--fused-top-terms", type=int, default=64)
    parser.add_argument("--fused-top-mbtags", type=int, default=32)
    parser.add_argument("--prior-smoothing", type=float, default=10.0)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def spark_path(path: Path) -> str:
    return path.resolve().as_uri()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="ascii") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def load_labels(spark: SparkSession, path: Path) -> DataFrame:
    labels = spark.read.parquet(spark_path(path)).select(*AUDIT_COLUMNS)
    invalid = labels.groupBy("artist_id").agg(F.countDistinct("split").alias("splits"))
    if invalid.where(F.col("splits") != 1).limit(1).count():
        raise ValueError("an artist appears in multiple splits")
    return labels.persist(StorageLevel.DISK_ONLY)


def clean_tags(spark: SparkSession, path: Path) -> DataFrame:
    return (
        spark.read.parquet(spark_path(path))
        .select(
            F.col("artist_id").cast("string"),
            F.lower(F.trim(F.col("tag"))).alias("tag"),
            F.col("source").cast("string"),
        )
        .where(
            F.col("artist_id").isNotNull()
            & (F.col("artist_id") != "")
            & F.col("tag").isNotNull()
            & (F.col("tag") != "")
            & F.col("source").isin("term", "mbtag")
        )
        .distinct()
        .persist(StorageLevel.DISK_ONLY)
    )


def choose_top_tags(
    tags: DataFrame,
    train_artists: DataFrame,
    source: str,
    count: int,
) -> list[str]:
    rows = (
        tags.where(F.col("source") == source)
        .join(F.broadcast(train_artists), "artist_id", "inner")
        .groupBy("tag")
        .agg(F.count("*").alias("artists"))
        .orderBy(F.desc("artists"), F.asc("tag"))
        .limit(count)
        .collect()
    )
    return [str(row["tag"]) for row in rows]


def build_indicators(
    tags: DataFrame,
    top_terms: list[str],
    top_mbtags: list[str],
) -> DataFrame:
    names = indicator_columns(len(top_terms), len(top_mbtags))
    pairs = [
        *(("term", tag, names[index]) for index, tag in enumerate(top_terms)),
        *(
            ("mbtag", tag, names[len(top_terms) + index])
            for index, tag in enumerate(top_mbtags)
        ),
    ]
    mapping = F.create_map(
        *(
            item
            for source, tag, name in pairs
            for item in (F.lit(f"{source}\u0000{tag}"), F.lit(name))
        )
    )
    selected = tags.withColumn(
        "indicator",
        F.element_at(mapping, F.concat_ws("\u0000", "source", "tag")),
    ).where(F.col("indicator").isNotNull())
    return selected.groupBy("artist_id").pivot("indicator", list(names)).agg(F.lit(1))


def tag_counts(tags: DataFrame) -> DataFrame:
    return tags.groupBy("artist_id").agg(
        F.sum(F.when(F.col("source") == "term", 1).otherwise(0)).cast("double").alias("term_count"),
        F.sum(F.when(F.col("source") == "mbtag", 1).otherwise(0))
        .cast("double")
        .alias("mbtag_count"),
        F.count("*").cast("double").alias("tag_count"),
    )


def tag_era_features(tags: DataFrame) -> DataFrame:
    long_year = F.regexp_extract(
        F.col("tag"), r"(?:^|[^0-9])((?:19[2-9]|20[01])0)s?(?:[^0-9]|$)", 1
    )
    short_year = F.regexp_extract(
        F.col("tag"), r"(?:^|[^0-9])((?:[2-9]0|00))s(?:[^0-9]|$)", 1
    )
    era = (
        F.when(long_year != "", long_year.cast("double"))
        .when(short_year == "00", F.lit(2000.0))
        .when(short_year != "", short_year.cast("double") + F.lit(1900.0))
    )
    return tags.withColumn("era", era).where(F.col("era").isNotNull()).groupBy(
        "artist_id"
    ).agg(
        F.count("*").cast("double").alias(ERA_COLUMNS[0]),
        F.avg("era").alias(ERA_COLUMNS[1]),
        F.stddev_pop("era").alias(ERA_COLUMNS[2]),
        F.min("era").alias(ERA_COLUMNS[3]),
        F.max("era").alias(ERA_COLUMNS[4]),
    )


def tag_year_priors(
    tags: DataFrame,
    artists: DataFrame,
    train_years: DataFrame,
    global_year: float,
    smoothing: float,
    source: str,
) -> DataFrame:
    source_tags = tags.where(F.col("source") == source)
    totals = source_tags.join(train_years, "artist_id", "inner").groupBy("tag").agg(
        F.sum("artist_year").alias("year_sum"),
        F.count("*").alias("artist_count"),
    )
    own = train_years.select("artist_id", F.col("artist_year").alias("own_year"))
    rows = (
        source_tags.join(artists, "artist_id", "inner")
        .join(totals, "tag", "left")
        .join(own, "artist_id", "left")
    )
    remove_own = (F.col("split") == "train") & F.col("own_year").isNotNull()
    support = F.col("artist_count") - F.when(remove_own, 1).otherwise(0)
    total = F.col("year_sum") - F.when(remove_own, F.col("own_year")).otherwise(0.0)
    prior = (total + F.lit(smoothing * global_year)) / (support + F.lit(smoothing))
    prefix = f"{source}_year_prior_"
    return rows.withColumn("support", support).withColumn("prior", prior).groupBy(
        "artist_id"
    ).agg(
        F.count("prior").cast("double").alias(prefix + "count"),
        F.avg("prior").alias(prefix + "mean"),
        F.stddev_pop("prior").alias(prefix + "std"),
        F.min("prior").alias(prefix + "min"),
        F.max("prior").alias(prefix + "max"),
        F.avg("support").alias(prefix + "support_mean"),
    )


def similarity_features(
    spark: SparkSession,
    path: Path,
    train_years: DataFrame,
) -> DataFrame:
    edges = (
        spark.read.parquet(spark_path(path))
        .select("artist_id", "similar_artist_id", "edge_order")
        .where(F.col("artist_id") != F.col("similar_artist_id"))
        .groupBy("artist_id", "similar_artist_id")
        .agg(F.min("edge_order").alias("edge_order"))
    )
    known = train_years.select(
        F.col("artist_id").alias("similar_artist_id"), "artist_year"
    )
    order = Window.partitionBy("artist_id").orderBy(
        F.asc("edge_order"), F.asc("similar_artist_id")
    )
    ranked = edges.join(known, "similar_artist_id", "inner").withColumn(
        "neighbor_rank", F.row_number().over(order)
    )
    top_k_expressions = [
        expression
        for size in SIMILARITY_TOP_K
        for expression in (
            F.count(F.when(F.col("neighbor_rank") <= size, 1)).cast("double").alias(
                f"similar_top_{size}_count"
            ),
            F.avg(
                F.when(F.col("neighbor_rank") <= size, F.col("artist_year"))
            ).alias(f"similar_top_{size}_year_mean"),
            F.stddev_pop(
                F.when(F.col("neighbor_rank") <= size, F.col("artist_year"))
            ).alias(f"similar_top_{size}_year_std"),
        )
    ]
    rank_expressions = [
        F.max(F.when(F.col("neighbor_rank") == rank, F.col("artist_year"))).alias(
            name
        )
        for rank, name in enumerate(GRAPH_RANK_COLUMNS, start=1)
    ]
    return ranked.groupBy("artist_id").agg(
        F.count("*").cast("double").alias(GRAPH_COLUMNS[0]),
        F.avg("artist_year").alias(GRAPH_COLUMNS[1]),
        F.stddev_pop("artist_year").alias(GRAPH_COLUMNS[2]),
        F.min("artist_year").alias(GRAPH_COLUMNS[3]),
        F.percentile_approx("artist_year", 0.10, 1000).alias(GRAPH_COLUMNS[4]),
        F.percentile_approx("artist_year", 0.50, 1000).alias(GRAPH_COLUMNS[5]),
        F.percentile_approx("artist_year", 0.90, 1000).alias(GRAPH_COLUMNS[6]),
        F.max("artist_year").alias(GRAPH_COLUMNS[7]),
        *top_k_expressions,
        *rank_expressions,
    )


def scalar_features(spark: SparkSession, path: Path, labels: DataFrame) -> DataFrame:
    scalar = spark.read.parquet(spark_path(path)).select(
        "track_id",
        F.col("artist_id").alias("scalar_artist_id"),
        *SCALAR_COLUMNS,
    )
    joined = labels.join(scalar, "track_id", "inner")
    if joined.where(F.col("artist_id") != F.col("scalar_artist_id")).limit(1).count():
        raise ValueError("scalar and label artist IDs disagree")
    return joined.select(
        *AUDIT_COLUMNS,
        *(
            F.col(name).cast("double").alias(name)
            for name in SCALAR_COLUMNS
        ),
        *(
            F.when(F.col(name).isNull() | F.isnan(name), 1.0).otherwise(0.0).alias(missing)
            for name, missing in zip(SCALAR_COLUMNS, SCALAR_MISSING_COLUMNS, strict=True)
        ),
    )


def build(args: argparse.Namespace, spark: SparkSession) -> dict[str, Any]:
    prepare_output(args.output, args.overwrite)
    labels = load_labels(spark, args.labels)
    artists = labels.select("artist_id", "split").distinct()
    train_years = labels.where(F.col("split") == "train").groupBy("artist_id").agg(
        F.avg("year").alias("artist_year")
    ).persist(StorageLevel.DISK_ONLY)
    global_year = float(labels.where(F.col("split") == "train").agg(F.avg("year")).first()[0])
    tags = clean_tags(spark, args.metadata / "artist_tags.parquet")
    train_artists = artists.where(F.col("split") == "train").select("artist_id")
    top_terms = choose_top_tags(tags, train_artists, "term", args.top_terms)
    top_mbtags = choose_top_tags(tags, train_artists, "mbtag", args.top_mbtags)
    indicators = build_indicators(tags, top_terms, top_mbtags)
    artist_features = (
        artists.join(tag_counts(tags), "artist_id", "left")
        .join(tag_era_features(tags), "artist_id", "left")
        .join(
            tag_year_priors(
                tags, artists, train_years, global_year, args.prior_smoothing, "term"
            ),
            "artist_id",
            "left",
        )
        .join(
            tag_year_priors(
                tags, artists, train_years, global_year, args.prior_smoothing, "mbtag"
            ),
            "artist_id",
            "left",
        )
        .join(
            similarity_features(
                spark, args.metadata / "artist_similarity.parquet", train_years
            ),
            "artist_id",
            "left",
        )
        .join(indicators, "artist_id", "left")
    )
    locations = spark.read.parquet(
        spark_path(args.metadata / "artist_location.parquet")
    ).select(
        "artist_id",
        F.col("latitude").cast("double").alias(LOCATION_COLUMNS[0]),
        F.col("longitude").cast("double").alias(LOCATION_COLUMNS[1]),
    )
    artist_features = artist_features.join(locations, "artist_id", "left").withColumn(
        LOCATION_COLUMNS[2],
        F.when(F.col(LOCATION_COLUMNS[0]).isNull(), 1.0).otherwise(0.0),
    )
    raise NotImplementedError("metadata assembly incomplete")
