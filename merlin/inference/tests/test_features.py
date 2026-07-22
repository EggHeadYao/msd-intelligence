from __future__ import annotations

import math
import unittest

from merlin.inference.features import (
    RANKER_V1_FEATURES,
    InferenceFeatureComputer,
    TrackMetadata,
    build_track_metadata,
)
from merlin.inference.types import Candidate


class InferenceFeaturesTest(unittest.TestCase):
    def test_computes_exact_ranker_v1_feature_contract(self):
        computer = InferenceFeatureComputer({
            "q": TrackMetadata("album", 2000, 0.4),
            "c": TrackMetadata("album", 2003, 0.8),
        })
        candidate = Candidate(
            "c",
            frozenset({"audio", "bfs", "tag"}),
            {"audio": 0.9, "bfs": 0.5, "tag": 0.7},
        )

        features = computer.compute("q", candidate)

        self.assertEqual(tuple(features), RANKER_V1_FEATURES)
        self.assertEqual(features, {
            "cos_audio": 0.9, "cos_graph": 0.0, "bfs_score": 0.5,
            "tag_tfidf_cosine": 0.7, "same_album": 1.0, "same_year": 0.0,
            "year_gap": 3.0, "candidate_popularity": 0.8,
            "from_audio": 1.0, "from_graph": 0.0,
            "from_bfs": 1.0, "from_tag": 1.0,
        })

    def test_missing_metadata_and_non_finite_scores_fall_back_to_zero(self):
        computer = InferenceFeatureComputer({})
        candidate = Candidate("unknown", frozenset({"graph"}), {"graph": math.nan})

        features = computer.compute("query", candidate)

        self.assertTrue(all(math.isfinite(value) for value in features.values()))
        self.assertEqual(features["cos_graph"], 0.0)
        self.assertEqual(features["from_graph"], 1.0)
        self.assertEqual(features["candidate_popularity"], 0.0)

    def test_build_metadata_applies_year_mask(self):
        tracks = build_track_metadata([
            ("q", "album", 2000, False, None),
            ("c", None, 2001, True, 0.5),
        ])

        self.assertIsNone(tracks["q"].year)
        self.assertEqual(tracks["c"].year, 2001)
        self.assertEqual(tracks["q"].popularity, 0.0)

    def test_build_metadata_rejects_conflicting_rows(self):
        with self.assertRaisesRegex(ValueError, "conflicting metadata"):
            build_track_metadata([
                ("q", "first", 2000, True, 0.5),
                ("q", "second", 2000, True, 0.5),
            ])
