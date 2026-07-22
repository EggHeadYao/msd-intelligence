from __future__ import annotations

import math
import unittest

from merlin.inference.feature_schema import RANKER_V2_FEATURES
from merlin.inference.features_v2 import (
    FeatureFillValues,
    PairSignalLookups,
    RankerV2FeatureComputer,
    TrackMetadataV2,
    build_track_metadata_v2,
)
from merlin.inference.types import Candidate


class RankerV2FeaturesTest(unittest.TestCase):

    def test_computes_signals_independently_of_recall_provenance(self):
        signals = PairSignalLookups(
            audio=lambda _q, _c: 0.8,
            graph=lambda _q, _c: 0.6,
            bfs=lambda _q, _c: 0.5,
            tags=lambda _q, _c: 0.25,
        )
        computer = RankerV2FeatureComputer(
            {"q": TrackMetadataV2("r1", 2000), "c": TrackMetadataV2("r1", 2003)},
            signals,
        )
        candidate = Candidate("c", sources=frozenset({"audio"}))

        features = computer.compute("q", candidate)

        self.assertEqual(tuple(features), RANKER_V2_FEATURES)
        self.assertEqual(features, {
            "cos_audio": 0.8, "cos_graph": 0.6, "has_graph": 1.0,
            "bfs_score": 0.5, "has_bfs": 1.0,
            "tag_tfidf_cosine": 0.25, "has_tags": 1.0,
            "same_release": 1.0, "has_release": 1.0,
            "year_gap": 3.0, "has_year": 1.0,
            "audio_tag_interaction": 0.2, "graph_bfs_interaction": 0.3,
        })

    def test_fills_missing_signals_and_keeps_availability_masks(self):
        signals = PairSignalLookups(
            audio=lambda _q, _c: 0.4,
            graph=lambda _q, _c: None,
            bfs=lambda _q, _c: math.nan,
            tags=lambda _q, _c: None,
        )
        fills = FeatureFillValues({
            "cos_graph": 0.2,
            "bfs_score": 0.1,
            "tag_tfidf_cosine": 0.3,
            "year_gap": 5.0,
        })
        computer = RankerV2FeatureComputer({}, signals, fills)

        features = computer.compute("q", Candidate("c"))

        self.assertEqual(features["has_graph"], 0.0)
        self.assertEqual(features["has_bfs"], 0.0)
        self.assertEqual(features["has_tags"], 0.0)
        self.assertEqual(features["has_release"], 0.0)
        self.assertEqual(features["has_year"], 0.0)
        self.assertEqual(features["year_gap"], 5.0)
        self.assertAlmostEqual(features["audio_tag_interaction"], 0.12)
        self.assertAlmostEqual(features["graph_bfs_interaction"], 0.02)
