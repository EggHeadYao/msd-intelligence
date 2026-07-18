from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "training"))

from ridge_math import (  # noqa: E402
    finite_difference_gradient,
    gradient_step,
    relative_error,
    ridge_gradient,
    ridge_loss,
)


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "ridge_oracle.json"


class RidgeMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with FIXTURE_PATH.open("r", encoding="ascii") as handle:
            cls.fixture = json.load(handle)


if __name__ == "__main__":
    unittest.main()
