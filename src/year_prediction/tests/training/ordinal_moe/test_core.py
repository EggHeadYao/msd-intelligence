from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "src" / "training" / "ordinal_moe"
sys.path.insert(0, str(MODULE))

from ordinal_moe_core import (  # noqa: E402
    LossConfig,
    ParameterLayout,
    adam_step,
    batch_gradient,
    forward,
    huber,
)


class OrdinalMoECoreTest(unittest.TestCase):
    def test_huber_has_quadratic_and_linear_regions(self):
        loss, gradient = huber(np.array([-4.0, -1.0, 0.0, 2.0]), 2.0)
        np.testing.assert_allclose(loss, [6.0, 0.5, 0.0, 2.0])
        np.testing.assert_allclose(gradient, [-2.0, -1.0, 0.0, 2.0])

    def test_forward_heads_are_finite_and_bounded(self):
        layout = ParameterLayout(3)
        parameters = np.zeros(layout.size)
        output = forward(np.array([[1.0, -2.0, 0.5]]), parameters, layout, LossConfig())
        for name in ("ordinal_year", "moe_year", "direct_year", "blend_year"):
            self.assertTrue(np.all(np.isfinite(output[name])))
            self.assertGreaterEqual(float(output[name][0]), 1922.0)
            self.assertLessEqual(float(output[name][0]), 2011.0)

    def test_batch_gradient_shape_and_optimizer_projection(self):
        layout = ParameterLayout(2)
        parameters = np.zeros(layout.size)
        features = np.array([[1.0, 0.0], [0.0, 1.0]])
        years = np.array([1970.0, 2000.0])
        gradient, losses, count = batch_gradient(
            features, years, parameters, layout, LossConfig()
        )
        self.assertEqual(gradient.shape, parameters.shape)
        self.assertEqual(losses.shape, (5,))
        self.assertEqual(count, 2)
        updated, _, _, norm = adam_step(
            parameters, gradient, np.zeros_like(parameters),
            np.zeros_like(parameters), 1, 0.001, 5.0,
        )
        thresholds = updated[layout.slices()["thresholds"]]
        self.assertTrue(np.all(np.diff(thresholds) >= 0.0))
        self.assertTrue(np.all(np.isfinite(updated)))
        self.assertGreater(norm, 0.0)


if __name__ == "__main__":
    unittest.main()
