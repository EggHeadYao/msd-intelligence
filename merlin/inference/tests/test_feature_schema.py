from __future__ import annotations

import json
from pathlib import Path
import unittest

from merlin.inference.feature_schema import (
    RANKER_V2_FEATURES,
    RANKER_V2_SCHEMA_VERSION,
)


class FeatureSchemaTest(unittest.TestCase):
    def test_example_artifact_matches_python_contract(self):
        artifact_path = Path(__file__).parents[1] / "ranker_artifact.example.json"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact["feature_schema_version"], RANKER_V2_SCHEMA_VERSION)
        self.assertEqual(tuple(artifact["feature_order"]), RANKER_V2_FEATURES)
        size = len(RANKER_V2_FEATURES)
        self.assertEqual(len(artifact["means"]), size)
        self.assertEqual(len(artifact["stds"]), size)
        self.assertEqual(len(artifact["coefficients"]), size)

    def test_provenance_and_popularity_are_not_ranker_features(self):
        forbidden = {"candidate_popularity", "from_audio", "from_graph", "from_bfs", "from_tag"}

        self.assertTrue(forbidden.isdisjoint(RANKER_V2_FEATURES))


if __name__ == "__main__":
    unittest.main()
