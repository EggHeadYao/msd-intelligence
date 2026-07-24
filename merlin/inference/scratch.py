"""Scratch-space guards shared by high-volume C3 Spark stages."""

import math
from pathlib import Path
import shutil


def estimate_ranker_scratch_gb(
    *,
    training_rows: int,
    validation_rows: int,
    feature_count: int,
    model_count: int,
) -> float:
    """Estimate the peak serialized vectors and ranking shuffle, not Parquet bytes."""
    values = (training_rows, feature_count, model_count)
    if any(value <= 0 for value in values) or validation_rows < 0:
        raise ValueError("ranker scratch estimate inputs are invalid")
    training_vectors = training_rows * (feature_count * 8 + 24) * 1.5
    validation_vectors = validation_rows * (feature_count * 8 + 96) * 1.5
    ranking_shuffle = validation_rows * model_count * 64 * 2
    projected_gib = max(training_vectors, validation_vectors + ranking_shuffle) / (1024**3)
    return math.ceil(projected_gib * 4) / 4


def prepare_scratch_root(
    root: str | Path,
    *,
    scope: str,
    min_free_gb: float | None,
    projected_gb: float = 0.0,
) -> Path:
    if scope not in {"formal", "smoke"}:
        raise ValueError("scratch scope must be formal or smoke")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    minimum = min_free_gb
    if minimum is None:
        minimum = 16.0 if scope == "formal" else 0.25
    if minimum < 0 or projected_gb < 0:
        raise ValueError("free-space requirements must be non-negative")
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    required_gb = minimum + projected_gb
    if free_gb < required_gb:
        raise OSError(
            f"C3 storage has {free_gb:.2f} GiB free; {required_gb:.2f} GiB required "
            f"({minimum:.2f} GiB reserve + {projected_gb:.2f} GiB projected work)"
        )
    return path
