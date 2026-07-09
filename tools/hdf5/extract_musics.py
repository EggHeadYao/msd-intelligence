# ruff: noqa: T201
"""Extract segment arrays, similar artists, and artist terms from per-song .h5 files."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


BATCH_SIZE: int = 10000


def aggregate_segments(
    pitches: np.ndarray,
    timbre: np.ndarray,
    loudness_max: np.ndarray,
) -> np.ndarray:
    """Aggregate variable-length segment arrays into a fixed 99-dim vector.

    Args:
        pitches: Nx12 float64 array from /analysis/segments_pitches.
        timbre: Nx12 float64 array from /analysis/segments_timbre.
        loudness_max: N float64 array from /analysis/segments_loudness_max.

    Returns:
        1-D numpy array of length 99 (48 + 48 + 3).
        Order: pitches_mean[12], pitches_std[12], pitches_min[12], pitches_max[12],
               timbre_mean[12],  timbre_std[12],  timbre_min[12],  timbre_max[12],
               loudness_mean, loudness_std, loudness_max_agg.

    """
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
        result.extend([0.0, 0.0, 0.0])
    else:
        result.append(float(np.mean(loudness_max)))
        result.append(float(np.std(loudness_max, ddof=0)))
        result.append(float(np.max(loudness_max)))
    return np.array(result, dtype=np.float64)


def process_one_file(
    path: Path,
) -> tuple[np.ndarray, list[tuple[str, str]], list[tuple[str, str]]]:
    """Extract aggregated features, similar artists, and terms from one .h5 file.

    Args:
        path: Path to a per-song HDF5 file.

    Returns:
        Tuple of (features_99, similar_pairs, term_pairs).
        features_99: 1-D float64 numpy array of 99 aggregated segment features.
        similar_pairs: List of (track_id, similar_artist_id) with empty strings
                       filtered out.
        term_pairs: List of (track_id, term) with empty strings filtered out.

    """
    with h5py.File(path, "r") as h5:
        track_id: str = _decode(h5["/analysis/songs"][0]["track_id"])

        features: np.ndarray = aggregate_segments(
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

    return features, similar_pairs, term_pairs


def _decode(val: object) -> str:
    """Decode HDF5 byte strings to Python str, pass through native str."""
    if isinstance(val, bytes | bytearray):
        return bytes(val).decode("utf-8")
    return str(val)


def discover_files(root: Path) -> list[Path]:
    """Walk a directory tree and return all .h5 file paths."""
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        files.extend(Path(dirpath) / name for name in filenames if name.endswith(".h5"))
    return files


def load_checkpoint(path: Path) -> set[str]:
    """Load the set of already-processed absolute file paths."""
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def save_checkpoint(path: Path, done: set[str]) -> None:
    """Persist the set of processed file paths."""
    path.write_text(json.dumps(sorted(done)))


def flush_batch(
    output_dir: Path,
    batch_idx: int,
    feats: list[np.ndarray],
    sims: list[tuple[str, str]],
    terms: list[tuple[str, str]],
) -> None:
    """Write one batch of accumulated data to Parquet files."""
    suffix: str = f"{batch_idx:04d}.parquet"
    pq.write_table(
        pa.table({"features": np.vstack(feats).tolist()}),
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
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <data_root> <output_dir>")
        sys.exit(1)

    data_root: Path = Path(sys.argv[1])
    output_dir: Path = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path: Path = output_dir / "checkpoint.json"

    all_files: list[Path] = discover_files(data_root)
    done: set[str] = load_checkpoint(checkpoint_path)
    remaining: list[Path] = [f for f in all_files if str(f) not in done]

    print(
        f"Found {len(all_files)} .h5 files, "
        f"{len(done)} done, {len(remaining)} remaining",
    )

    feats_batch: list[np.ndarray] = []
    sim_batch: list[tuple[str, str]] = []
    term_batch: list[tuple[str, str]] = []
    batch_idx: int = 0

    for path in remaining:
        try:
            feats, sims, terms = process_one_file(path)
        except OSError:
            print(f"ERROR processing {path}", file=sys.stderr)
            continue

        feats_batch.append(feats)
        sim_batch.extend(sims)
        term_batch.extend(terms)
        done.add(str(path))

        if len(feats_batch) >= BATCH_SIZE:
            flush_batch(output_dir, batch_idx, feats_batch, sim_batch, term_batch)
            save_checkpoint(checkpoint_path, done)
            batch_idx += 1
            feats_batch.clear()
            sim_batch.clear()
            term_batch.clear()

    if feats_batch:
        flush_batch(output_dir, batch_idx, feats_batch, sim_batch, term_batch)
        save_checkpoint(checkpoint_path, done)

    print(f"Done. Processed {len(done)} files total.")


if __name__ == "__main__":
    main()
