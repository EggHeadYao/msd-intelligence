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

    def test_golden_values(self) -> None:
        fixture = self.fixture
        expected = fixture["expected"]
        loss = ridge_loss(
            fixture["features"],
            fixture["labels"],
            fixture["weights"],
            fixture["intercept"],
            fixture["l2"],
        )
        gradient, intercept_gradient = ridge_gradient(
            fixture["features"],
            fixture["labels"],
            fixture["weights"],
            fixture["intercept"],
            fixture["l2"],
        )
        updated_weights, updated_intercept = gradient_step(
            fixture["weights"],
            fixture["intercept"],
            gradient,
            intercept_gradient,
            fixture["learning_rate"],
        )
        self.assertAlmostEqual(loss, expected["loss"], places=12)
        self.assertLess(relative_error(gradient, expected["gradient"]), 1e-12)
        self.assertAlmostEqual(
            intercept_gradient, expected["intercept_gradient"], places=12
        )
        self.assertLess(
            relative_error(updated_weights, expected["updated_weights"]), 1e-12
        )
        self.assertAlmostEqual(
            updated_intercept, expected["updated_intercept"], places=12
        )



if __name__ == "__main__":
    unittest.main()
