from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from merlin.inference.ranker import LogisticRanker


class LogisticRankerTest(unittest.TestCase):
    def test_loads_matching_split_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {"feature_schema_version": "v2", "feature_order": ["x"]}
            schema = root / "schema.json"
            scaler = root / "scaler.json"
            model = root / "model.json"
            schema.write_text(json.dumps(common), encoding="utf-8")
            scaler.write_text(
                json.dumps({**common, "means": [1], "stds": [2]}), encoding="utf-8"
            )
            model.write_text(json.dumps({
                **common, "model_type": "logistic_regression",
                "coefficients": [3], "intercept": 0.5,
            }), encoding="utf-8")

            ranker = LogisticRanker.from_artifacts(schema, scaler, model)

        self.assertEqual(ranker.score({"x": 5}), 6.5)

    def test_rejects_split_artifact_order_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schema = root / "schema.json"
            scaler = root / "scaler.json"
            model = root / "model.json"
            schema.write_text(json.dumps({
                "feature_schema_version": "v2", "feature_order": ["x"],
            }), encoding="utf-8")
            scaler.write_text(json.dumps({
                "feature_schema_version": "v2", "feature_order": ["wrong"],
            }), encoding="utf-8")
            model.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "feature order mismatch"):
                LogisticRanker.from_artifacts(schema, scaler, model)

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
