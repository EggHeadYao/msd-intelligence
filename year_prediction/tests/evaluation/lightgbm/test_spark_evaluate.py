from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "src" / "evaluation" / "lightgbm"
sys.path.insert(0, str(MODULE))

from spark_evaluate import evaluate  # noqa: E402


class SparkLightGBMEvaluationTest(unittest.TestCase):
    def test_invalid_partition_count_is_rejected_before_io(self):
        args = argparse.Namespace(
            model_root=Path("model"), input=Path("data"),
            manifest=Path("manifest.json"), output=Path("output"),
            partitions=0, max_rows=None, overwrite=False,
        )
        with self.assertRaisesRegex(ValueError, "partitions must be positive"):
            evaluate(args, None)


if __name__ == "__main__":
    unittest.main()
