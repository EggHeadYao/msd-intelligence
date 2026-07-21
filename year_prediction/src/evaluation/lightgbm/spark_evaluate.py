from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession
from synapse.ml.lightgbm import LightGBMRegressionModel

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parents[1] / "training"
LIGHTGBM_DIR = TRAINING_DIR / "lightgbm"
sys.path[:0] = [str(TRAINING_DIR), str(LIGHTGBM_DIR)]

from lightgbm_metrics import (  # noqa: E402
    add_prediction_columns,
    decade_metrics,
    regression_metrics,
)
from spark_common import (  # noqa: E402
    load_feature_contract,
    load_feature_frame,
    prepare_output,
    split_counts,
    write_json,
)
from spark_io import write_parquet_parts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Spark LightGBM on test data")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def evaluate(args: argparse.Namespace, spark: SparkSession) -> dict:
    if args.partitions <= 0:
        raise ValueError("partitions must be positive")
    prepare_output(args.output, args.overwrite)
    started = time.perf_counter()
    contract = load_feature_contract(args.manifest)
    saved_contract = read_json(args.model_root / "feature_contract.json")
    if saved_contract["predictor_order_sha256"] != contract.order_sha256:
        raise ValueError("model and input feature order differ")
    frame = load_feature_frame(
        spark, args.input, contract, args.max_rows, ("test",)
    ).repartition(args.partitions)
    counts = split_counts(frame)
    if counts.get("test", 0) <= 0:
        raise ValueError("test split is empty")
    model_text = (args.model_root / "model.txt").read_text(encoding="ascii")
    model = LightGBMRegressionModel.loadNativeModelFromString(model_text)
    predictions = add_prediction_columns(model.transform(frame)).select(
        "track_id",
        "artist_id",
        "year",
        "split",
