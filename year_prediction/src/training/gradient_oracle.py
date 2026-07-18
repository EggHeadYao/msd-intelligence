from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pyspark import SparkContext
from pyspark.sql import SparkSession

from ridge_math import (
    finite_difference_gradient,
    gradient_step,
    relative_error,
    ridge_gradient,
    ridge_loss,
)


Partial = tuple[float, list[float], float, int]


def default_fixture_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "ridge_oracle.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Ridge gradient contract.")
    parser.add_argument("--fixture", type=Path, default=default_fixture_path())
    return parser.parse_args()


def load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def ship_oracle_modules(context: SparkContext) -> None:
    module_dir = Path(__file__).resolve().parent
    context.addPyFile(str(module_dir / "ridge_math.py"))
    context.addPyFile(str(module_dir / "gradient_oracle.py"))

