from __future__ import annotations

import tempfile
from pathlib import Path

from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType


SCHEMA = StructType(
    [
        StructField("track_id", StringType(), False),
        StructField("artist_id", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("normalized_year", DoubleType(), False),
        StructField("features", ArrayType(DoubleType(), False), False),
        StructField("split", StringType(), False),
    ]
)


def rows():
    definitions = (
        (1930, [-1.0, 0.2, 0.0, 0.1], "train"),
        (1940, [-0.8, -0.2, 0.1, 0.0], "train"),
        (1950, [-0.5, 0.3, -0.1, 0.2], "train"),
        (1960, [-0.2, -0.1, 0.2, -0.1], "train"),
        (1980, [0.2, 0.1, -0.2, 0.1], "train"),
        (1990, [0.5, -0.3, 0.1, -0.2], "train"),
        (2000, [0.8, 0.2, 0.0, 0.1], "train"),
        (2010, [1.0, -0.2, -0.1, 0.0], "train"),
        (1970, [0.0, 0.1, 0.0, -0.1], "validation"),
        (1985, [0.4, -0.1, 0.1, 0.0], "validation"),
        (1955, [-0.4, 0.0, -0.1, 0.1], "test"),
        (1995, [0.6, 0.0, 0.1, -0.1], "test"),
    )
    return [
        (f"TR{i:04d}", f"AR{i:04d}", year, (year - 1922.0) / 89.0, features, split)
        for i, (year, features, split) in enumerate(definitions)
    ]


def exercise(spark, representation: str, loss: str):
    from kernel_sgd.evaluator import evaluate
    from kernel_sgd.runner_impl import train
    from spark_io import write_parquet_parts

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "vectors.parquet"
        write_parquet_parts(spark.createDataFrame(rows(), SCHEMA), source)
        config = {
            "model_id": f"integration-{representation}-{loss}", "input": str(source),
            "output_root": str(root / "models"), "representation": representation, "loss": loss,
            "pca_variance_threshold": 0.9, "rff_dimension": 8, "gamma": 0.5, "seed": 472,
            "huber_delta_years": 5.0, "max_iterations": 3, "learning_rate": 0.1,
            "l2": 0.001, "gradient_tolerance": 0.0, "validation_interval": 1,
            "early_stopping_patience": 0, "early_stopping_min_delta": 0.0,
        }
        model_root = train(config, spark)
        test_root = evaluate(model_root, source, root / "results", spark)
        assert (model_root / "model.json").is_file()
        assert (model_root / "transform.npz").is_file()
        assert (test_root / "metrics.json").is_file()
        parts = [str(path) for path in (test_root / "predictions.parquet").glob("*.parquet")]
        assert spark.read.parquet(*parts).count() == 2
