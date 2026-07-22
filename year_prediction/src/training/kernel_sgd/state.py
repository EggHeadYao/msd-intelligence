from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyspark.ml.feature import PCA
from pyspark.ml.functions import array_to_vector
from pyspark.sql import DataFrame

from .features import Transform, pca_transform, rff_transform


def minimum_dimension(explained: np.ndarray, threshold: float) -> int:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("PCA variance threshold must be in (0, 1]")
    cumulative = np.cumsum(np.asarray(explained, dtype=np.float64))
    matches = np.flatnonzero(cumulative >= threshold)
    if not matches.size:
        raise ValueError("PCA did not reach the variance threshold")
    return int(matches[0] + 1)


def fit_transform(frame: DataFrame, config: dict) -> tuple[Transform, dict]:
    dimension = int(frame.selectExpr("size(features) AS n").first()["n"])
    representation = config["representation"]
    if representation == "pca":
        train = frame.where("split = 'train'").withColumn("vector", array_to_vector("features"))
        model = PCA(k=dimension, inputCol="vector", outputCol="pca").fit(train)
        explained = np.asarray(model.explainedVariance.toArray(), dtype=np.float64)
        output_dimension = minimum_dimension(explained, float(config["pca_variance_threshold"]))
        transform = pca_transform(np.asarray(model.pc.toArray())[:, :output_dimension])
        metadata = {
            "kind": "pca",
            "fit_split": "train",
            "input_dimension": dimension,
            "output_dimension": output_dimension,
            "variance_threshold": float(config["pca_variance_threshold"]),
            "cumulative_explained_variance": float(explained[:output_dimension].sum()),
        }
    else:
        transform = rff_transform(
            dimension, int(config["rff_dimension"]), float(config["gamma"]), int(config["seed"])
        )
        metadata = {
            "kind": "rff",
            "fit_split": "train",
            "input_dimension": dimension,
            "output_dimension": transform.output_dimension,
            "gamma": float(config["gamma"]),
            "seed": int(config["seed"]),
            "concatenate_input": representation == "t90_rff",
        }
    return transform, metadata


def save_transform(output: Path, transform: Transform, metadata: dict) -> None:
    np.savez_compressed(output / "transform.npz", matrix=transform.matrix, offset=transform.offset)
    (output / "transform.json").write_text(json.dumps(metadata, indent=2), encoding="ascii")


def load_transform(output: Path) -> tuple[Transform, dict]:
    metadata = json.loads((output / "transform.json").read_text(encoding="ascii"))
    with np.load(output / "transform.npz") as values:
        matrix, offset = np.asarray(values["matrix"]), np.asarray(values["offset"])
    transform = Transform(
        metadata["kind"], int(metadata["input_dimension"]),
        int(metadata["output_dimension"]), matrix, offset
    )
    return transform, metadata
