from __future__ import annotations

import sys

from pyspark import cloudpickle

cloudpickle.register_pickle_by_value(sys.modules[__name__])

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from pyspark import RDD
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

AUDIT_COLUMNS = ("track_id", "artist_id", "year", "split")
CATEGORICAL_COLUMNS = ("key", "mode", "time_signature")
CONTRACT_VERSION = "year_prediction_features_v1"
MIN_YEAR = 1922
MAX_YEAR = 2011


@dataclass(frozen=True)
class FeatureContract:
    predictors: tuple[str, ...]
    order_sha256: str
    expected_splits: dict[str, int]

    @property
    def dimension(self) -> int:
        return len(self.predictors)

    @property
    def categorical_indexes(self) -> tuple[int, ...]:
        return tuple(self.predictors.index(name) for name in CATEGORICAL_COLUMNS)


@dataclass(frozen=True)
class Standardization:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    finite_counts: tuple[int, ...]
    row_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "means": list(self.means),
            "scales": list(self.scales),
            "finite_counts": list(self.finite_counts),
            "row_count": self.row_count,
        }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(path)

