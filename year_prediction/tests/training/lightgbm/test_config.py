from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[3] / "src" / "training" / "lightgbm"
sys.path.insert(0, str(MODULE))

from lightgbm_config import file_sha256, read_config  # noqa: E402


class LightGBMConfigTest(unittest.TestCase):
    def test_paths_are_converted_and_hash_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"input": "data", "num_iterations": 10}), "ascii")
            loaded = read_config(path, {"input", "num_iterations"})
            self.assertEqual(loaded, {"input": Path("data"), "num_iterations": 10})
            self.assertEqual(file_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_unknown_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"unknown": 1}', "ascii")
            with self.assertRaisesRegex(ValueError, "unknown"):
                read_config(path, {"input"})


if __name__ == "__main__":
    unittest.main()
