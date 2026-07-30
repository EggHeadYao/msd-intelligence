from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[3] / "src" / "training"
sys.path.insert(0, str(TRAINING))

from spark_common import load_feature_contract  # noqa: E402


def order_hash(columns: list[str]) -> str:
    payload = json.dumps(columns, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class FeatureViewTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.manifest = Path(self.directory.name) / "manifest.json"
        full = ["key", "mode", "time_signature", "feature"]
        t90 = ["t90_0", "t90_1"]
        self.manifest.write_text(
            json.dumps(
                {
                    "contract_version": "year_prediction_features_v1",
                    "views": {
                        "full_tabular": {
                            "predictor_count": len(full),
                            "predictor_columns": full,
                            "predictor_order_sha256": order_hash(full),
                        },
                        "t90": {
                            "predictor_count": len(t90),
                            "predictor_columns": t90,
                            "predictor_order_sha256": order_hash(t90),
                        },
                    },
                    "counts": {
                        "splits": {
                            name: {"tracks": count}
                            for name, count in (
                                ("train", 3),
                                ("validation", 2),
                                ("test", 1),
                            )
                        }
                    },
                }
            ),
            encoding="ascii",
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_default_view_keeps_categorical_indexes(self):
        contract = load_feature_contract(self.manifest)
        self.assertEqual(contract.view_name, "full_tabular")
        self.assertEqual(contract.categorical_indexes, (0, 1, 2))

    def test_t90_view_has_no_categorical_indexes(self):
        contract = load_feature_contract(self.manifest, "t90")
        self.assertEqual(contract.dimension, 2)
        self.assertEqual(contract.categorical_indexes, ())

    def test_unknown_view_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown feature view"):
            load_feature_contract(self.manifest, "missing")
