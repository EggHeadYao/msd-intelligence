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

