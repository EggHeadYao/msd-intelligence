from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parents[2] / "src" / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

from metrics import (  # noqa: E402
    finalize_metric_partial,
    merge_metric_partials,
    prediction_metric_partial,
)


class RegressionMetricsTest(unittest.TestCase):
    def test_metrics_use_continuous_clipped_year_predictions(self):
        rows = [
            prediction_metric_partial(0.0, -0.1),
            prediction_metric_partial(0.5, 0.75),
            prediction_metric_partial(1.0, 1.2),
        ]
        total = rows[0]
        for row in rows[1:]:
            total = merge_metric_partials(total, row)
        metrics = finalize_metric_partial(total)
        self.assertEqual(metrics.count, 3)
        self.assertAlmostEqual(metrics.mae_years, 89.0 * 0.25 / 3.0)
        self.assertAlmostEqual(metrics.raw_out_of_range_rate, 2.0 / 3.0)
        self.assertGreater(metrics.raw_mae_years, metrics.mae_years)
        self.assertTrue(math.isfinite(metrics.rmse_years))


if __name__ == "__main__":
    unittest.main()
