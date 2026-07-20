from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

RIDGE_DIR = Path(__file__).resolve().parents[3] / "src" / "training" / "ridge"
sys.path.insert(0, str(RIDGE_DIR))

from optimizer import gradient_norm, gradient_step  # noqa: E402


class RidgeOptimizerTest(unittest.TestCase):
    def test_gradient_norm_includes_intercept(self):
        self.assertAlmostEqual(gradient_norm([3.0, 4.0], 12.0), 13.0)

    def test_gradient_step_updates_weights_and_intercept(self):
        weights, intercept = gradient_step(
            [1.0, -2.0],
            0.5,
            [0.25, -0.5],
            -0.75,
            0.2,
        )
        self.assertEqual(weights, [0.95, -1.9])
        self.assertAlmostEqual(intercept, 0.65)

    def test_gradient_step_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            gradient_step([1.0], 0.0, [1.0, 2.0], 0.0, 0.1)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            gradient_step([1.0], 0.0, [1.0], 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "inputs must be finite"):
            gradient_step([1.0], 0.0, [math.inf], 0.0, 0.1)


if __name__ == "__main__":
    unittest.main()
