from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "src" / "features"
sys.path.insert(0, str(FEATURE_DIR))

from contract import (  # noqa: E402
    AUDIT_COLUMNS,
    DERIVED_SCALAR_COLUMNS,
    GLOBAL_SCALAR_COLUMNS,
    TRACK_ID,
)
from full_tabular import build_full_tabular  # noqa: E402

