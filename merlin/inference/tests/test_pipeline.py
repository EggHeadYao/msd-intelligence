from __future__ import annotations

import unittest

from merlin.inference.pipeline import MerlinPipeline
from merlin.inference.ranker import LogisticRanker
from merlin.inference.retrieval import VectorRetriever


class _Features:
    schema_version = "test-v1"

    def compute(self, query_track_id, candidate):
        return {"score": candidate.recall_scores["audio"]}


class _NoRedundancy:
    def similarity(self, left_track_id, right_track_id):
        return 0.0


class PipelineTest(unittest.TestCase):
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
            redundancy=_NoRedundancy(),
            ranker_limit=2,
            final_limit=2,
        )

        result = pipeline.recommend("query")

        self.assertEqual([item.track_id for item in result], ["best", "other"])
        self.assertEqual([item.rank for item in result], [1, 2])


if __name__ == "__main__":
    unittest.main()
