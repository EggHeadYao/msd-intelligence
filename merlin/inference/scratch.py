"""Scratch-space guards shared by high-volume C3 Spark stages."""

from pathlib import Path
import shutil


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
            f"({minimum:.2f} GiB reserve + {projected_gb:.2f} GiB projected output)"
        )
    return path
