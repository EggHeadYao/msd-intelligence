import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parents[3] / "src" / "training"
sys.path.insert(0, str(TRAINING))

from kernel_sgd.objective import residual_terms  # noqa: E402


class HuberObjectiveTest(unittest.TestCase):
    def test_large_residual_has_bounded_gradient(self):
        losses, derivatives = residual_terms(np.asarray([-3.0, 0.5, 4.0]), "huber", 1.0)
        np.testing.assert_allclose(derivatives, [-1.0, 0.5, 1.0])
        np.testing.assert_allclose(losses, [2.5, 0.125, 3.5])

    def test_squared_loss_remains_available(self):
        losses, derivatives = residual_terms(np.asarray([2.0]), "squared", 1.0)
        np.testing.assert_allclose(losses, [2.0])
        np.testing.assert_allclose(derivatives, [2.0])


if __name__ == "__main__":
    unittest.main()
