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
