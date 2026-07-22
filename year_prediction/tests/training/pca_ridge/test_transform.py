import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parents[3] / "src" / "training"
sys.path.insert(0, str(TRAINING))

from kernel_sgd.features import apply_transform, pca_transform  # noqa: E402
from kernel_sgd.state import minimum_dimension  # noqa: E402


class PcaTransformTest(unittest.TestCase):
    def test_threshold_and_projection(self):
        self.assertEqual(minimum_dimension(np.asarray([0.7, 0.2, 0.1]), 0.85), 2)
        transform = pca_transform(np.asarray([[1.0, 0.0], [0.0, 2.0]]))
        actual = apply_transform(np.asarray([3.0, 4.0]), transform)
        np.testing.assert_allclose(actual, [3.0, 8.0])


if __name__ == "__main__":
    unittest.main()
