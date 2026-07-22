from __future__ import annotations

import unittest

from merlin.inference.pipeline import MerlinPipeline
from merlin.inference.ranker import LogisticRanker
from merlin.inference.retrieval import VectorRetriever


class _Features:
    schema_version = "test-v1"

    def compute(self, query_track_id, candidate):
        return {"score": candidate.recall_scores["audio"]}


class PipelineTest(unittest.TestCase):
    def test_recall_reports_source_shortage_and_unique_count(self):
        audio = VectorRetriever("audio", lambda _query, _limit: [("one", 0.8)])
        pipeline = MerlinPipeline(
            retrievers=[audio],
            retriever_limits={"audio": 3},
            feature_computer=_Features(),
            ranker=LogisticRanker.mock("test-v1", ["score"]),
            final_limit=2,
        )

        candidates, audit = pipeline.recall("query")

        self.assertEqual([item.track_id for item in candidates], ["one"])
        self.assertEqual(audit.source_counts, {"audio": 1})
        self.assertEqual(audit.source_shortages, {"audio": 2})
        self.assertEqual(audit.unique_candidates, 1)

    def test_vector_recall_overfetches_and_filters_same_song(self):
        requested = []

        def search(_query, limit):
            requested.append(limit)
            return [
                ("query", 1.0), ("sibling", 0.9),
                ("best", 0.8), ("best", 0.7), ("other", 0.6),
            ]

        retriever = VectorRetriever(
            "audio",
            search,
            same_song=lambda _query, candidate: candidate == "sibling",
        )

        candidates = retriever.retrieve("query", limit=2)

        self.assertEqual(requested, [7])
        self.assertEqual([item.track_id for item in candidates], ["best", "other"])
        self.assertEqual([item.source_ranks["audio"] for item in candidates], [1, 2])

    def test_recommend_excludes_query_and_sorts_candidates(self):
        audio = VectorRetriever(
            "audio",
            lambda _query, _limit: [("query", 1.0), ("best", 0.9), ("other", 0.2)],
        )
        ranker = LogisticRanker.mock("test-v1", ["score"])
        pipeline = MerlinPipeline(
            retrievers=[audio],
            retriever_limits={"audio": 3},
            feature_computer=_Features(),
            ranker=ranker,
            final_limit=2,
        )

        result = pipeline.recommend("query")

        self.assertEqual([item.track_id for item in result], ["best", "other"])
        self.assertEqual([item.rank for item in result], [1, 2])
