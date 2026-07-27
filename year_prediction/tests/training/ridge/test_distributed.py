from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = ROOT / "src" / "training"
RIDGE_DIR = TRAINING_DIR / "ridge"
EVALUATION_DIR = ROOT / "src" / "evaluation"
ORACLE_DIR = ROOT / "tests" / "oracles" / "ridge"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(RIDGE_DIR))
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(ORACLE_DIR))

from distributed import (  # noqa: E402
    direct_batch_statistics,
    direct_full_batch_statistics,
    sample_mini_batch,
)
from reference import ridge_gradient, ridge_loss  # noqa: E402
from train import validate_config  # noqa: E402


class DistributedRidgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("DistributedRidgeTest")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")
        for path in (
            TRAINING_DIR / "target.py",
            EVALUATION_DIR / "metrics.py",
            RIDGE_DIR / "objectives.py",
            RIDGE_DIR / "distributed.py",
        ):
            cls.spark.sparkContext.addPyFile(str(path))
        with (ORACLE_DIR / "fixture.json").open("r", encoding="ascii") as handle:
            cls.fixture = json.load(handle)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_partitioned_statistics_match_local_oracle(self):
        fixture = self.fixture
        expected_gradient, expected_intercept = ridge_gradient(
            fixture["features"],
            fixture["labels"],
            fixture["weights"],
            fixture["intercept"],
            fixture["l2"],
        )
        expected_loss = ridge_loss(
            fixture["features"],
            fixture["labels"],
            fixture["weights"],
            fixture["intercept"],
            fixture["l2"],
        )
        points = list(zip(fixture["features"], fixture["labels"]))
        for partitions in (1, 2, 4):
            rdd = self.spark.sparkContext.parallelize(points, partitions)
            actual = direct_full_batch_statistics(
                rdd, fixture["weights"], fixture["intercept"], fixture["l2"]
            )
            self.assertAlmostEqual(actual.objective, expected_loss, places=10)
            for value, expected in zip(actual.gradient, expected_gradient):
                self.assertAlmostEqual(value, expected, places=10)
            self.assertAlmostEqual(actual.intercept_gradient, expected_intercept, places=10)

    def test_mini_batch_sampling_is_reproducible(self):
        points = [([float(index)], float(index % 2)) for index in range(1000)]
        rdd = self.spark.sparkContext.parallelize(points, 4)
        first = sample_mini_batch(rdd, 0.25, 472).collect()
        repeated = sample_mini_batch(rdd, 0.25, 472).collect()
        different = sample_mini_batch(rdd, 0.25, 473).collect()
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        self.assertGreater(len(first), 150)
        self.assertLess(len(first), 350)

    def test_mini_batch_sampling_rejects_full_fraction(self):
        rdd = self.spark.sparkContext.parallelize([([1.0], 0.0)])
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            sample_mini_batch(rdd, 1.0, 472)

    def test_mini_batch_configs_are_valid(self):
        for name in ("ridge_t90_minibatch_25.json", "ridge_t90_minibatch_10.json"):
            with self.subTest(config=name):
                with (ROOT / "config" / name).open(encoding="ascii") as handle:
                    validate_config(json.load(handle))

    def test_direct_batch_statistics_accepts_sampled_points(self):
        points = self.spark.sparkContext.parallelize([([1.0], 0.0), ([2.0], 1.0)])
        sampled = sample_mini_batch(points, 0.9, 472)
        statistics = direct_batch_statistics(sampled, [0.0], 0.0, 0.0)
        self.assertGreater(statistics.count, 0)


if __name__ == "__main__":
    unittest.main()
