from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


EXPECTED_SONGS = 1_000_000
EMBEDDING_COLUMN = "embedding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MERLIN C1 PCA audio outputs.")
    parser.add_argument("--output", type=Path, default=Path("parquets/merlin/audio"))
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_SONGS)
    parser.add_argument("--norm-tolerance", type=float, default=1e-6)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    return parser.parse_args()
