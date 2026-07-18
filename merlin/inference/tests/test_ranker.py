from __future__ import annotations

import unittest

from merlin.inference.ranker import LogisticRanker


class LogisticRankerTest(unittest.TestCase):
    def test_returns_hand_computed_standardized_raw_margin(self):
        ranker = LogisticRanker(
            feature_schema_version="test-v2",
            feature_order=("left", "right"),
            means=(1.0, 4.0),
            stds=(2.0, 2.0),
            coefficients=(3.0, -1.0),
            intercept=0.5,
        )

        margin = ranker.score({"left": 5.0, "right": 2.0})

        self.assertEqual(margin, 7.5)
        self.assertEqual(margin, ranker.raw_margin({"left": 5.0, "right": 2.0}))

    def test_display_score_is_sigmoid_but_not_primary_score(self):
        ranker = LogisticRanker.mock("test-v2", ["signal"])

        margin = ranker.score({"signal": 2.0})
        display = ranker.display_score({"signal": 2.0})

        self.assertEqual(margin, 2.0)
        self.assertGreater(display, 0.5)
        self.assertLess(display, 1.0)


if __name__ == "__main__":
    unittest.main()
