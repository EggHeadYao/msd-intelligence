from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = ROOT / "src" / "training"
RIDGE_DIR = TRAINING_DIR / "ridge"
ORACLE_DIR = ROOT / "tests" / "oracles" / "ridge"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(RIDGE_DIR))
sys.path.insert(0, str(ORACLE_DIR))

from objectives import (  # noqa: E402
    finalize_ridge_partial,
    merge_ridge_partials,
    squared_point_partial,
)
from optimizer import gradient_norm, gradient_step  # noqa: E402
from reference import ridge_gradient, ridge_loss  # noqa: E402


class RidgeObjectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ORACLE_DIR / "fixture.json").open("r", encoding="ascii") as handle:
            cls.fixture = json.load(handle)

    def test_production_statistics_match_oracle(self):
        fixture = self.fixture
        partials = [
            squared_point_partial(point, fixture["weights"], fixture["intercept"])
            for point in zip(fixture["features"], fixture["labels"])
        ]
        merged = partials[0]
        for partial in partials[1:]:
            merged = merge_ridge_partials(merged, partial)
        statistics = finalize_ridge_partial(merged, fixture["weights"], fixture["l2"])
        expected_gradient, expected_intercept = ridge_gradient(
            fixture["features"],
            fixture["labels"],
            fixture["weights"],
            fixture["intercept"],
            fixture["l2"],
        )
        self.assertAlmostEqual(
            statistics.objective,
            ridge_loss(
                fixture["features"],
                fixture["labels"],
                fixture["weights"],
                fixture["intercept"],
                fixture["l2"],
            ),
            places=12,
        )
        for actual, expected in zip(statistics.gradient, expected_gradient):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(statistics.intercept_gradient, expected_intercept, places=12)
        self.assertEqual(statistics.count, len(fixture["features"]))

    def test_update_matches_golden_fixture(self):
        fixture = self.fixture
        weights, intercept = gradient_step(
            fixture["weights"],
            fixture["intercept"],
            fixture["expected"]["gradient"],
            fixture["expected"]["intercept_gradient"],
            fixture["learning_rate"],
        )
        for actual, expected in zip(weights, fixture["expected"]["updated_weights"]):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(intercept, fixture["expected"]["updated_intercept"], places=12)
        self.assertTrue(math.isfinite(gradient_norm(weights, intercept)))


if __name__ == "__main__":
    unittest.main()
