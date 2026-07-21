from __future__ import annotations

import unittest

import numpy as np

from faiss_checks import validate_source_mapping


class FakeIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype(np.float32)
        self.d = int(vectors.shape[1])

    def reconstruct(self, row_id: int) -> np.ndarray:
        return self.vectors[row_id]


class FaissSourceMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vectors = np.eye(4, dtype=np.float32)
        self.index = FakeIndex(self.vectors)

    def test_exact_mapping_passes(self) -> None:
        queries = [(row_id, f"track-{row_id}", self.vectors[row_id]) for row_id in range(4)]
        _, error = validate_source_mapping(self.index, queries)
        self.assertEqual(error, 0.0)

    def test_swapped_mapping_fails(self) -> None:
        queries = [(0, "track-1", self.vectors[1])]
        with self.assertRaisesRegex(AssertionError, "row mapping"):
            validate_source_mapping(self.index, queries)

    def test_non_unit_query_fails(self) -> None:
        queries = [(0, "track-0", np.zeros(4, dtype=np.float32))]
        with self.assertRaisesRegex(AssertionError, "unit normalized"):
            validate_source_mapping(self.index, queries)


if __name__ == "__main__":
    unittest.main()
