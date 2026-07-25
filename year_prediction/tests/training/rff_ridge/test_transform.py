import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parents[3] / "src" / "training"
sys.path.insert(0, str(TRAINING))

from kernel_sgd.features import (  # noqa: E402
    apply_transform,
    rff_transform,
    transform_partition,
)


class RffRidgeTransformTest(unittest.TestCase):
    def test_seed_is_deterministic_and_input_is_concatenated(self):
        first = rff_transform(2, 4, 0.5, 472)
        second = rff_transform(2, 4, 0.5, 472)
        np.testing.assert_allclose(first.matrix, second.matrix)
        values = apply_transform(np.asarray([1.0, -1.0]), first, concatenate=True)
        self.assertEqual(values.shape, (6,))
        np.testing.assert_allclose(values[:2], [1.0, -1.0])

    def test_batched_transform_matches_individual_transform(self):
        transform = rff_transform(2, 4, 0.5, 472)
        rows = [
            (f"TR{i}", f"AR{i}", 1980 + i, 0.5, np.asarray([i, -i], dtype=np.float64))
            for i in range(5)
        ]
        actual = list(transform_partition(rows, transform, concatenate=True, batch_size=2))
        for source, transformed in zip(rows, actual):
            self.assertEqual(transformed[:4], source[:4])
            np.testing.assert_allclose(
                transformed[4], apply_transform(source[4], transform, concatenate=True)
            )


if __name__ == "__main__":
    unittest.main()
