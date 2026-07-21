from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

MODULE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from ordinal_moe_core import LossConfig, ParameterLayout  # noqa: E402
from ordinal_moe_train import evaluate_heads, prediction_frame  # noqa: E402
from spark_io import write_parquet_parts  # noqa: E402
from spark_common import (  # noqa: E402
    Standardization,
    load_feature_contract,
    load_feature_frame,
    point_rdd,
    prepare_output,
    standardize_frame,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Spark Ordinal-MoE inference")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("parquets/year_prediction/features/full_tabular.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("parquets/year_prediction/features/manifest.json"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def run(args: argparse.Namespace, spark: SparkSession) -> dict:
    prepare_output(args.output, args.overwrite)
    contract = load_feature_contract(args.manifest)
    metadata = read_json(args.model_root / "model.json")
    if metadata["feature_order_sha256"] != contract.order_sha256:
        raise ValueError("model and feature contract hashes differ")
    layout = ParameterLayout(int(metadata["dimension"]))
    if layout.dimension != contract.dimension:
        raise ValueError("model and feature dimensions differ")
    config = LossConfig(**metadata["loss_config"])
    config.validate()
    with np.load(args.model_root / "model.npz") as payload:
        parameters = np.asarray(payload["parameters"], dtype=np.float64)
