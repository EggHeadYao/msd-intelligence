from __future__ import annotations

import unittest

from merlin.inference.retrieval import TagRetriever
from merlin.inference.tag_data import build_tag_data, find_similar_artists


class TagRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.data = build_tag_data(
            [("q", "a"), ("b1", "b"), ("b2", "b"), ("c1", "c")],
            [
                ("a", "Rock"),
                ("a", "jazz"),
                ("b", "rock"),
                ("b", "jazz"),
                ("c", "rock"),
                ("c", "rock"),
            ],
        )

    def test_builds_normalized_bidirectional_tag_mappings(self):
        self.assertEqual(self.data.artist_terms["a"], {"rock", "jazz"})
        self.assertEqual(self.data.term_artists["rock"], {"a", "b", "c"})
        self.assertEqual(self.data.artist_tracks["b"], ["b1", "b2"])

    def test_ranks_exact_tag_match_before_partial_match(self):
        neighbors = find_similar_artists(self.data, "a", top_k=2)

        self.assertEqual([artist for artist, _ in neighbors], ["b", "c"])
        self.assertAlmostEqual(neighbors[0][1], 1.0)
        self.assertGreater(neighbors[0][1], neighbors[1][1])

    def test_ignores_tags_above_frequency_cap(self):
        neighbors = find_similar_artists(
            self.data, "a", top_k=2, max_term_artists=2
        )

        self.assertEqual([artist for artist, _ in neighbors], ["b"])

    def test_retriever_accepts_lazy_neighbor_search(self):
        retriever = TagRetriever(
            self.data.track_to_artist,
            lambda artist: find_similar_artists(self.data, artist, 2),
            self.data.artist_tracks,
            per_artist_cap=1,
        )

        candidates = retriever.retrieve("q", limit=2)

        self.assertEqual([item.track_id for item in candidates], ["b1", "c1"])
        self.assertEqual([item.source_ranks["tag"] for item in candidates], [1, 2])
