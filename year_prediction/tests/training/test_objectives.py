from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "src" / "training"
ORACLE_DIR = ROOT / "tests" / "oracles" / "ridge"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(ORACLE_DIR))

from objectives import (  # noqa: E402
    finalize_ridge_partial,
    merge_ridge_partials,
    squared_point_partial,
)
from optimizer import gradient_norm, gradient_step  # noqa: E402
from reference import ridge_gradient, ridge_loss  # noqa: E402


