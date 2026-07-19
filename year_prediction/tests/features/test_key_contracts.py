from __future__ import annotations

import sys
import unittest
from pathlib import Path


FEATURES_DIR = Path(__file__).resolve().parents[2] / "src" / "features"
sys.path.insert(0, str(FEATURES_DIR))

from columns import candidate_columns  # noqa: E402
from key_contracts import (  # noqa: E402
    K0,
    K1,
    K2,
    K3,
    KEY_CONTRACTS,
    KEY_COS_COLUMN,
    KEY_SIN_COLUMN,
    KEY_UNKNOWN_COLUMN,
    key_contract_metadata,
    key_feature_columns,
    require_key_contract,
)


class KeyContractsTest(unittest.TestCase):
    def test_contract_columns_have_stable_order(self):
        expected = {
            K0: (),
            K1: (*(f"key_{value}" for value in range(12)), KEY_UNKNOWN_COLUMN),
            K2: (KEY_SIN_COLUMN, KEY_COS_COLUMN, KEY_UNKNOWN_COLUMN),
            K3: (KEY_SIN_COLUMN, KEY_COS_COLUMN, KEY_UNKNOWN_COLUMN),
        }
        for contract in KEY_CONTRACTS:
            self.assertEqual(key_feature_columns(contract), expected[contract])
            candidates = candidate_columns(contract, (3, 4))
            self.assertEqual(len(candidates), len(set(candidates)))
            self.assertTrue(set(expected[contract]).issubset(candidates))

    def test_metadata_identifies_each_encoding(self):
        encodings = {
            K0: "no_key",
            K1: "one_hot",
            K2: "chromatic_circle",
            K3: "circle_of_fifths",
        }
        for contract, encoding in encodings.items():
            metadata = key_contract_metadata(contract)
            self.assertEqual(metadata["id"], contract)
            self.assertEqual(metadata["encoding"], encoding)
            self.assertEqual(metadata["columns"], list(key_feature_columns(contract)))

    def test_invalid_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported key contract"):
            require_key_contract("k4")


if __name__ == "__main__":
    unittest.main()
