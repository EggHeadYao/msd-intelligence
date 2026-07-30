from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "src" / "evaluation" / "ordinal_moe"
sys.path.insert(0, str(MODULE))

from evaluate import evaluation_args  # noqa: E402


class OrdinalMoEEvaluationTest(unittest.TestCase):
    def test_evaluation_is_locked_to_test(self):
        args = argparse.Namespace(model_root=Path("model"), input=Path("data"))
        locked = evaluation_args(args)
        self.assertEqual(locked.split, "test")
        self.assertEqual(locked.model_root, Path("model"))


if __name__ == "__main__":
    unittest.main()
