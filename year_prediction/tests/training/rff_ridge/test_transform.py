import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parents[3] / "src" / "training"
sys.path.insert(0, str(TRAINING))

from kernel_sgd.features import apply_transform, rff_transform  # noqa: E402


class RffRidgeTransformTest(unittest.TestCase):
    def test_seed_is_deterministic_and_input_is_concatenated(self):
        first = rff_transform(2, 4, 0.5, 472)
        second = rff_transform(2, 4, 0.5, 472)
        np.testing.assert_allclose(first.matrix, second.matrix)
        values = apply_transform(np.asarray([1.0, -1.0]), first, concatenate=True)
        self.assertEqual(values.shape, (6,))
        np.testing.assert_allclose(values[:2], [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
