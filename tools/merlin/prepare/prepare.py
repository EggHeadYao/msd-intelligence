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

