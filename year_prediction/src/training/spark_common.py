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


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _order_sha256(columns: Iterable[str]) -> str:
    encoded = json.dumps(list(columns), ensure_ascii=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_feature_contract(path: Path) -> FeatureContract:
    with path.open("r", encoding="ascii") as handle:
        manifest = json.load(handle)
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("unexpected feature contract version")
    view = manifest.get("views", {}).get("full_tabular", {})
    predictors = tuple(view.get("predictor_columns", ()))
    expected_hash = str(view.get("predictor_order_sha256", ""))
    if len(predictors) != 594 or len(set(predictors)) != len(predictors):
        raise ValueError("full-tabular predictor contract is invalid")
    if _order_sha256(predictors) != expected_hash:
        raise ValueError("full-tabular predictor order hash is invalid")
    if not set(CATEGORICAL_COLUMNS).issubset(predictors):
        raise ValueError("categorical predictors are missing")
    split_payload = manifest.get("counts", {}).get("splits", {})
    split_counts = {
        name: int(split_payload[name]["tracks"])
        for name in ("train", "validation", "test")
    }
    return FeatureContract(predictors, expected_hash, split_counts)


def _clean_predictor(name: str):
    value = F.col(name).cast("double")
    invalid = value.isNull() | F.isnan(value) | (F.abs(value) == F.lit(float("inf")))
    if name == "tempo":
        invalid = invalid | (value <= F.lit(0.0))
    if name == "time_signature":
        invalid = invalid | (value <= F.lit(0.0))
    return F.when(invalid, F.lit(None).cast("double")).otherwise(value).alias(name)


def parquet_inputs(path: Path) -> list[str]:
    if path.is_dir():
        files = sorted(
            item
            for item in path.rglob("*.parquet")
            if item.is_file() and not item.name.startswith((".", "_"))
        )
        if not files:
            raise FileNotFoundError(f"no Parquet part files found under {path}")
        return [str(item) for item in files]
    return [str(path)]

def load_feature_frame(
    spark: SparkSession,
    input_path: Path,
    contract: FeatureContract,
    max_rows_per_split: int | None = None,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> DataFrame:
    source = spark.read.parquet(*parquet_inputs(input_path))
    expected = (*AUDIT_COLUMNS, *contract.predictors)
    missing = [name for name in expected if name not in source.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing[:5]}")
    selected = source.select(
        F.col("track_id").cast("string"),
        F.col("artist_id").cast("string"),
        F.col("year").cast("int"),
        F.col("split").cast("string"),
        *(_clean_predictor(name) for name in contract.predictors),
    ).where(F.col("split").isin(*splits))
    if max_rows_per_split is not None:
        if max_rows_per_split <= 0:
            raise ValueError("max_rows_per_split must be positive")
        parts = [
            selected.where(F.col("split") == split).limit(max_rows_per_split)
            for split in splits
        ]
        selected = parts[0]
        for part in parts[1:]:
            selected = selected.unionByName(part)
    assembler = VectorAssembler(
        inputCols=list(contract.predictors),
        outputCol="features",
        handleInvalid="keep",
    )
    return assembler.transform(selected).select(
        "track_id",
        "artist_id",
        "year",
        "split",
        F.col("year").cast("double").alias("label"),
        "features",
    )


def assert_artist_isolation(frame: DataFrame) -> None:
    artists = {
        name: frame.where(F.col("split") == name).select("artist_id").distinct()
        for name in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if artists[left].intersect(artists[right]).limit(1).count():
            raise ValueError(f"artist overlap between {left} and {right}")


def split_counts(frame: DataFrame) -> dict[str, int]:
    rows = frame.groupBy("split").count().collect()
    return {str(row["split"]): int(row["count"]) for row in rows}


StatsPartial = tuple[np.ndarray, np.ndarray, np.ndarray, int]


def _stats_partition(rows: Iterator[Row]) -> Iterator[StatsPartial]:
    sums = squares = counts = None
    row_count = 0
    for row in rows:
        values = np.asarray(row["features"].toArray(), dtype=np.float64)
        if sums is None:
            sums = np.zeros(values.size, dtype=np.float64)
            squares = np.zeros(values.size, dtype=np.float64)
            counts = np.zeros(values.size, dtype=np.int64)
        finite = np.isfinite(values)
        clean = np.where(finite, values, 0.0)
        sums += clean
        squares += clean * clean
        counts += finite
        row_count += 1
    if sums is not None and squares is not None and counts is not None:
        yield sums, squares, counts, row_count


def _merge_stats(left: StatsPartial, right: StatsPartial) -> StatsPartial:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2], left[3] + right[3]


def fit_standardization(train: DataFrame) -> Standardization:
    sums, squares, counts, row_count = train.select("features").rdd.mapPartitions(
        _stats_partition
    ).treeReduce(_merge_stats)
    safe_counts = np.maximum(counts, 1)
    means = sums / safe_counts
    variances = np.maximum(squares / safe_counts - means * means, 0.0)
    scales = np.sqrt(variances)
    means = np.where(counts > 0, means, 0.0)
    scales = np.where((counts > 1) & (scales > 1.0e-12), scales, 1.0)
    return Standardization(
        tuple(float(value) for value in means),
        tuple(float(value) for value in scales),
        tuple(int(value) for value in counts),
        int(row_count),
    )


STANDARDIZED_SCHEMA = StructType(
    [
        StructField("track_id", StringType(), False),
        StructField("artist_id", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("split", StringType(), False),
        StructField("label", DoubleType(), False),
        StructField("features", VectorUDT(), False),
    ]
)


def _standardize_partition(
    rows: Iterator[Row], means: np.ndarray, scales: np.ndarray, clip: float
) -> Iterator[tuple[str, str, int, str, float, Any]]:
    for row in rows:
        values = np.asarray(row["features"].toArray(), dtype=np.float64)
        values = np.where(np.isfinite(values), values, means)
        values = np.clip((values - means) / scales, -clip, clip)
        yield (
            str(row["track_id"]),
            str(row["artist_id"]),
            int(row["year"]),
            str(row["split"]),
            float(row["label"]),
            Vectors.dense(values),
        )


def standardize_frame(
    spark: SparkSession,
    frame: DataFrame,
    state: Standardization,
    clip: float = 10.0,
) -> DataFrame:
    means = np.asarray(state.means, dtype=np.float64)
    scales = np.asarray(state.scales, dtype=np.float64)
    values = frame.rdd.mapPartitions(
        lambda rows: _standardize_partition(rows, means, scales, clip)
    )
    return spark.createDataFrame(values, STANDARDIZED_SCHEMA)


def point_rdd(frame: DataFrame) -> RDD[tuple[str, str, int, np.ndarray]]:
    return frame.select("track_id", "artist_id", "year", "features").rdd.map(
        lambda row: (
            str(row["track_id"]),
