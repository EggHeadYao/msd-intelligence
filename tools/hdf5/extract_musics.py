# ruff: noqa: T201
"""Extract segment arrays, similar artists, and artist terms from per-song .h5 files."""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BATCH_SIZE: int = 10000


def _make_feature_columns() -> list[str]:
    """Return the 100 column names matching aggregate_segments output order."""
    cols: list[str] = []
    for prefix in ("pitch", "timbre"):
        for stat in ("mean", "std", "min", "max"):
            cols.extend(f"{prefix}_{stat}_{i}" for i in range(12))
    cols.extend(f"loudness_{stat}" for stat in ("mean", "std", "min", "max"))
    return cols


FEATURE_COLUMNS: list[str] = _make_feature_columns()
# 12 * 2 * 4 + 4 = 100 feature columns (+ has_segments in flush_batch = 101 total)


def aggregate_segments(
    pitches: np.ndarray,
    timbre: np.ndarray,
    loudness_max: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Aggregate variable-length segment arrays into a fixed 100-dim vector.

    Args:
        pitches: Nx12 float64 array from /analysis/segments_pitches.
        timbre: Nx12 float64 array from /analysis/segments_timbre.
        loudness_max: N float64 array from /analysis/segments_loudness_max.

    Returns:
        Tuple of (features_100, has_segments).
        features_100: 1-D float64 numpy array of length 100.
        has_segments: True if segments data was present (N > 0).

    """
    has_segments: bool = pitches.size > 0 or timbre.size > 0
    result: list[float] = []
    for arr in (pitches, timbre):
        if arr.size == 0:
            result.extend([0.0] * 48)
            continue
        result.extend(float(x) for x in np.mean(arr, axis=0))
        result.extend(float(x) for x in np.std(arr, axis=0, ddof=0))
        result.extend(float(x) for x in np.min(arr, axis=0))
        result.extend(float(x) for x in np.max(arr, axis=0))
    if loudness_max.size == 0:
        result.extend([0.0] * 4)
    else:
        result.append(float(np.mean(loudness_max)))
        result.append(float(np.std(loudness_max, ddof=0)))
        result.append(float(np.min(loudness_max)))
        result.append(float(np.max(loudness_max)))
    return np.array(result, dtype=np.float64), has_segments


def process_one_file(
    path: Path,
) -> tuple[np.ndarray, bool, list[tuple[str, str]], list[tuple[str, str]]]:
    """Extract aggregated features, similar artists, and terms from one .h5 file.

    Args:
        path: Path to a per-song HDF5 file.

    Returns:
        Tuple of (features_100, has_segments, similar_pairs, term_pairs).
        features_100: 1-D float64 numpy array of 100 aggregated segment features.
        has_segments: True if segments data was present.
        similar_pairs: List of (track_id, similar_artist_id) with empty strings
                       filtered out.
        term_pairs: List of (track_id, term) with empty strings filtered out.

    """
    with h5py.File(path, "r") as h5:
        track_id: str = _decode(h5["/analysis/songs"][0]["track_id"])

        features: np.ndarray
        has_segments_flag: bool
        features, has_segments_flag = aggregate_segments(
            h5["/analysis/segments_pitches"][:],
            h5["/analysis/segments_timbre"][:],
            h5["/analysis/segments_loudness_max"][:],
        )

        raw_similar: np.ndarray = h5["/metadata/similar_artists"][:]
        similar_pairs: list[tuple[str, str]] = [
            (track_id, _decode(aid)) for aid in raw_similar if _decode(aid)
        ]

        raw_terms: np.ndarray = h5["/metadata/artist_terms"][:]
        term_pairs: list[tuple[str, str]] = [
            (track_id, _decode(t)) for t in raw_terms if _decode(t)
        ]

    return features, has_segments_flag, similar_pairs, term_pairs


def _decode(val: object) -> str:
    """Decode HDF5 byte strings to Python str, pass through native str."""
    if isinstance(val, bytes | bytearray):
        return val.decode("utf-8")
    return str(val)


def discover_files(root: Path) -> list[Path]:
    """Walk a directory tree and return all .h5 file paths."""
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        files.extend(Path(dirpath) / name for name in filenames if name.endswith(".h5"))
    return files


def load_checkpoint(path: Path) -> set[str]:
    """Load already-processed file paths from a line-delimited file."""
    if not path.exists():
        return set()
    return {line.rstrip("\n") for line in path.read_text().splitlines() if line}


def save_checkpoint(path: Path, batch: list[str]) -> None:
    """Append a batch of processed file paths to the checkpoint file."""
    with path.open("a") as f:
        for p in batch:
            f.write(p + "\n")


def worker(
    path: Path,
) -> tuple[
    str,
    tuple[np.ndarray, bool, list[tuple[str, str]], list[tuple[str, str]]] | None,
]:
    """Call process_one_file in a worker process, catching OSError."""
    try:
        return str(path), process_one_file(path)
    except OSError:
        return str(path), None


def flush_batch(  # noqa: PLR0913
    output_dir: Path,
    batch_idx: int,
    feats: list[np.ndarray],
    seg_flags: list[bool],
    sims: list[tuple[str, str]],
    terms: list[tuple[str, str]],
) -> None:
    """Write one batch of accumulated data to Parquet files."""
    suffix: str = f"{batch_idx:04d}.parquet"
    mat: np.ndarray = np.vstack(feats)
    cols: dict[str, list[float]] = {
        name: mat[:, i].tolist() for i, name in enumerate(FEATURE_COLUMNS)
    }
    cols["has_segments"] = [int(f) for f in seg_flags]
    pq.write_table(
        pa.table(cols),
        str(output_dir / f"features_{suffix}"),
    )
    pq.write_table(
        pa.table(
            {
                "track_id": [s[0] for s in sims],
                "similar_artist_id": [s[1] for s in sims],
            },
        ),
        str(output_dir / f"similar_{suffix}"),
    )
    pq.write_table(
        pa.table(
            {
                "track_id": [t[0] for t in terms],
                "term": [t[1] for t in terms],
            },
        ),
        str(output_dir / f"terms_{suffix}"),
    )


def main() -> None:
    """Entry point: walk data root, process each .h5, write batch Parquet files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Root of HDF5 file tree")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path: Path = output_dir / "checkpoint.json"

    all_files: list[Path] = discover_files(data_root)
    done: set[str] = load_checkpoint(checkpoint_path)
    remaining: list[Path] = [f for f in all_files if str(f) not in done]

    workers: int = args.workers
    print(
        f"Found {len(all_files)} .h5 files, "
        f"{len(done)} done, {len(remaining)} remaining"
        f"{f', using {workers} workers' if workers > 1 else ''}",
    )

    feats_batch: list[np.ndarray] = []
    seg_flags_batch: list[bool] = []
    sim_batch: list[tuple[str, str]] = []
    term_batch: list[tuple[str, str]] = []
    done_batch: list[str] = []
    batch_idx: int = 0

    source = (
        Pool(workers).imap_unordered(worker, remaining)
        if workers > 1
        else (worker(p) for p in remaining)
    )

    for path_str, result in source:
        if result is None:
            print(f"ERROR processing {path_str}", file=sys.stderr)
            continue

        feats, has_seg, sims, terms = result
        feats_batch.append(feats)
        seg_flags_batch.append(has_seg)
        sim_batch.extend(sims)
        term_batch.extend(terms)
        done.add(path_str)
        done_batch.append(path_str)
        sys.stdout.write(f"\r[{len(done)}/{len(all_files)}]")
        sys.stdout.flush()

        if len(feats_batch) >= BATCH_SIZE:
            flush_batch(
                output_dir,
                batch_idx,
                feats_batch,
                seg_flags_batch,
                sim_batch,
                term_batch,
            )
            save_checkpoint(checkpoint_path, done_batch)
            batch_idx += 1
            feats_batch.clear()
            seg_flags_batch.clear()
            sim_batch.clear()
            term_batch.clear()
            done_batch.clear()

    if feats_batch:
        flush_batch(
            output_dir,
            batch_idx,
            feats_batch,
            seg_flags_batch,
            sim_batch,
            term_batch,
        )
        save_checkpoint(checkpoint_path, done_batch)

    print(f"Done. Processed {len(done)} files total.")


if __name__ == "__main__":
    main()
