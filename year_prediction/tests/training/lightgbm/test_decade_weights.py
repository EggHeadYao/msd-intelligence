from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[3] / "src" / "training" / "lightgbm"
sys.path.insert(0, str(MODULE))

from lightgbm_train import normalized_decade_weights  # noqa: E402


class DecadeWeightTest(unittest.TestCase):
    def test_weights_favor_rare_decades_and_keep_unit_mean(self):
        counts = {1950: 100, 2000: 900}
        weights = normalized_decade_weights(counts, 0.5, 3.0)
        mean = sum(counts[key] * weights[key] for key in counts) / sum(counts.values())
        self.assertGreater(weights[1950], weights[2000])
        self.assertAlmostEqual(mean, 1.0)

    def test_maximum_caps_the_relative_weight(self):
        weights = normalized_decade_weights({1930: 1, 2000: 10000}, 1.0, 2.0)
        self.assertLessEqual(weights[1930] / weights[2000], 2.0)
