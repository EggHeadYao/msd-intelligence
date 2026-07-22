from __future__ import annotations

import unittest

import numpy as np

from merlin.inference.faiss_index import FaissTrackIndex


class _FakeFlatIpIndex:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ntotal, self.d = self.vectors.shape

    def reconstruct(self, row_id):
        return self.vectors[row_id].copy()

    def search(self, queries, limit):
        scores = queries @ self.vectors.T
        row_ids = np.argsort(-scores, axis=1)[:, :limit]
        ordered_scores = np.take_along_axis(scores, row_ids, axis=1)
        return ordered_scores, row_ids


class FaissTrackIndexTest(unittest.TestCase):
    def test_search_maps_faiss_rows_back_to_tracks(self):
        index = FaissTrackIndex(
            _FakeFlatIpIndex([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
            ("query", "near", "far"),
        )

        results = index.search("query", 3)

        self.assertEqual([track_id for track_id, _ in results], ["query", "near", "far"])
        self.assertAlmostEqual(results[1][1], 0.8, places=6)

    def test_rejects_mapping_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "size"):
            FaissTrackIndex(_FakeFlatIpIndex([[1.0], [0.5]]), ("only-one",))

    def test_rejects_duplicate_track_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            FaissTrackIndex(_FakeFlatIpIndex([[1.0], [0.5]]), ("same", "same"))

    def test_rejects_unknown_query(self):
        index = FaissTrackIndex(_FakeFlatIpIndex([[1.0]]), ("known",))

        with self.assertRaisesRegex(KeyError, "not in FAISS mapping"):
            index.search("unknown", 1)
