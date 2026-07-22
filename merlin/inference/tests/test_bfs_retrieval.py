from __future__ import annotations

import unittest
from unittest.mock import patch

from merlin.inference.bfs_data import BfsData, build_bfs_data
from merlin.inference.retrieval import BfsRetriever


class BfsRetrievalTest(unittest.TestCase):
    @patch("merlin.inference.bfs_data.load_bfs_data")
    def test_constructs_retriever_from_parquet(self, load_data):
        load_data.return_value = BfsData({"q": "a"}, {"a": ["q"]}, {"a": ["b"]})

        retriever = BfsRetriever.from_parquet(
            "songs.parquet", "edges.parquet", max_depth=3, per_artist_cap=4
        )

        load_data.assert_called_once_with("songs.parquet", "edges.parquet")
        self.assertEqual(retriever.track_to_artist, {"q": "a"})
        self.assertEqual(retriever.max_depth, 3)
        self.assertEqual(retriever.per_artist_cap, 4)

    def test_builds_deduplicated_mappings(self):
        data = build_bfs_data(
            [("q", "a"), ("x", "b"), ("x", "b"), ("", "b")],
            [("a", "b"), ("a", "b"), ("a", "a"), ("b", "c")],
        )

        self.assertEqual(data.track_to_artist, {"q": "a", "x": "b"})
        self.assertEqual(data.artist_tracks, {"a": ["q"], "b": ["x"]})
        self.assertEqual(data.artist_neighbors, {"a": ["b"], "b": ["c"]})

    def test_retrieves_tracks_in_bfs_distance_order(self):
        data = build_bfs_data(
            [("q", "a"), ("b1", "b"), ("b2", "b"), ("c1", "c")],
            [("a", "b"), ("b", "c")],
        )
        retriever = BfsRetriever(
            data.track_to_artist,
            data.artist_neighbors,
            data.artist_tracks,
            max_depth=2,
            per_artist_cap=1,
        )

        candidates = retriever.retrieve("q", limit=10)

        self.assertEqual([item.track_id for item in candidates], ["b1", "c1"])
        self.assertEqual([item.recall_scores["bfs"] for item in candidates], [0.5, 1 / 3])
        self.assertEqual([item.source_ranks["bfs"] for item in candidates], [1, 2])

    def test_orders_equal_distance_by_tags_then_track_id(self):
        data = build_bfs_data(
            [("q", "a"), ("b2", "b"), ("b1", "b"), ("c1", "c")],
            [("a", "b"), ("a", "c")],
        )
        retriever = BfsRetriever(
            data.track_to_artist,
            data.artist_neighbors,
            data.artist_tracks,
            same_song=lambda _query, track: track == "b2",
            tag_similarity=lambda _root, artist: {"b": 0.8, "c": 0.4}[artist],
        )

        candidates = retriever.retrieve("q", limit=10)

        self.assertEqual([item.track_id for item in candidates], ["b1", "c1"])

    def test_rejects_conflicting_track_artist_rows(self):
        with self.assertRaisesRegex(ValueError, "multiple artists"):
            build_bfs_data([("q", "a"), ("q", "b")], [])
