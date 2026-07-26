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
