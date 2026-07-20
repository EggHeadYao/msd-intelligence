from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "src" / "features"
sys.path.insert(0, str(FEATURE_DIR))

from contract import (  # noqa: E402
    EXPECTED_GROUP_COUNTS,
    FEATURE_GROUPS,
    GLOBAL_SCALAR_COLUMNS,
    DERIVED_SCALAR_COLUMNS,
    T90_COLUMNS,
    YEAR_EXCLUDED_COLUMNS,
    YEAR_SHARED_FEATURE_SET,
    order_sha256,
)


class FeatureContractTest(unittest.TestCase):
    def test_dimensions_and_groups_are_disjoint(self):
        self.assertEqual(len(T90_COLUMNS), 90)
        self.assertEqual(len(YEAR_EXCLUDED_COLUMNS), 48)
        self.assertEqual(len(YEAR_SHARED_FEATURE_SET), 580)
        self.assertEqual(len(GLOBAL_SCALAR_COLUMNS + DERIVED_SCALAR_COLUMNS), 14)
        self.assertEqual(
            {name: len(columns) for name, columns in FEATURE_GROUPS.items()},
            EXPECTED_GROUP_COUNTS,
        )
        flattened = [column for columns in FEATURE_GROUPS.values() for column in columns]
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_t90_covariance_uses_diagonal_offset_order(self):
        self.assertEqual(T90_COLUMNS[12:15], (
            "t90_timbre_cov_0_0",
            "t90_timbre_cov_1_1",
            "t90_timbre_cov_2_2",
        ))
        self.assertEqual(T90_COLUMNS[-2:], (
            "t90_timbre_cov_1_11",
            "t90_timbre_cov_0_11",
        ))

    def test_order_hash_is_deterministic_and_order_sensitive(self):
        self.assertEqual(order_sha256(("a", "b")), order_sha256(("a", "b")))
        self.assertNotEqual(order_sha256(("a", "b")), order_sha256(("b", "a")))


if __name__ == "__main__":
    unittest.main()
