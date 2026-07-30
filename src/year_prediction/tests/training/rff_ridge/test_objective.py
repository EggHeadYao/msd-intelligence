import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parents[3] / "src" / "training"
sys.path.insert(0, str(TRAINING))

from kernel_sgd.objective import partition_statistics  # noqa: E402


class PartitionStatisticsTest(unittest.TestCase):
    def test_vectorized_statistics_match_direct_sums(self):
        weights = np.asarray([0.25, -0.5])
        intercept = 0.1
        rows = [
            ("T1", "A1", 1980, 0.2, np.asarray([1.0, 2.0])),
            ("T2", "A2", 1990, 0.6, np.asarray([-1.0, 0.5])),
            ("T3", "A3", 2000, 0.8, np.asarray([0.2, -0.4])),
        ]
        result = list(partition_statistics(rows, weights, intercept, "squared", 1.0))[0]
        features = np.stack([row[4] for row in rows])
        residuals = features @ weights + intercept - np.asarray([row[3] for row in rows])
        self.assertEqual(result.count, len(rows))
        self.assertAlmostEqual(result.loss_sum, float(0.5 * np.dot(residuals, residuals)))
        np.testing.assert_allclose(result.gradient_sum, features.T @ residuals)
        self.assertAlmostEqual(result.intercept_gradient_sum, float(residuals.sum()))


if __name__ == "__main__":
    unittest.main()
