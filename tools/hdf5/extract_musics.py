# ruff: noqa: T201
"""Extract fixed-width audio features from per-track MSD HDF5 files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.hdf5.audio_features import (
    CONTRACT_VERSION,
    FEATURE_COLUMNS,
    FEATURE_COUNT,
    aggregate_audio_features,
)

BATCH_SIZE = 10000
CONTRACT_NAME = "feature_contract.json"
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("track_id", pa.string(), nullable=False),
        *(pa.field(column, pa.float64()) for column in FEATURE_COLUMNS),
    ],
)


@dataclass(frozen=True)
class ExtractionResult:
    discovered: int
    completed: int
    processed: int
    failed: int
    batches_written: int


def _decode(value: object) -> str:
    if isinstance(value, bytes | bytearray | np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def process_one_file(path: Path) -> tuple[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        song = h5["/analysis/songs"][0]
        track_id = _decode(song["track_id"])
        arrays = {
            name: h5[f"/analysis/{name}"][:]
            for name in (
                "segments_start",
                "segments_confidence",
                "segments_pitches",
                "segments_timbre",
                "segments_loudness_start",
                "segments_loudness_max",
                "segments_loudness_max_time",
                "beats_start",
                "beats_confidence",
                "bars_start",
                "bars_confidence",
                "tatums_start",
                "tatums_confidence",
                "sections_start",
                "sections_confidence",
            )
        }
        return track_id, aggregate_audio_features(song, arrays)


def discover_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        files.extend(
            Path(directory) / name for name in sorted(filenames) if name.endswith(".h5")
        )
    return files


def load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line for line in path.read_text(encoding="utf-8").splitlines() if line}


def save_checkpoint(path: Path, completed: set[str]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        "".join(f"{item}\n" for item in sorted(completed)),
        encoding="utf-8",
    )
    temporary.replace(path)


def _existing_batches(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("features_*.parquet"))


def _contract_payload(data_root: Path) -> dict[str, object]:
    encoded_columns = json.dumps(
        FEATURE_COLUMNS,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "contract_version": CONTRACT_VERSION,
        "data_root": str(data_root.resolve()),
        "feature_count": FEATURE_COUNT,
        "feature_order_sha256": hashlib.sha256(encoded_columns).hexdigest(),
        "columns": ["track_id", *FEATURE_COLUMNS],
    }


def ensure_contract(data_root: Path, output_dir: Path) -> None:
    path = output_dir / CONTRACT_NAME
    expected = _contract_payload(data_root)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(f"Existing {CONTRACT_NAME} does not match this run")
        return
    if _existing_batches(output_dir):
        raise RuntimeError(f"Existing batches have no {CONTRACT_NAME}")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def recover_completed(
    output_dir: Path,
    file_by_track_id: dict[str, str],
) -> set[str]:
    checkpoint_path = output_dir / "checkpoint.txt"
    checkpoint = load_checkpoint(checkpoint_path)
    known_files = set(file_by_track_id.values())
    unknown_checkpoint = checkpoint - known_files
    if unknown_checkpoint:
        raise RuntimeError(
            f"Checkpoint contains {len(unknown_checkpoint)} paths outside data_root",
        )

    recovered: set[str] = set()
    seen_tracks: set[str] = set()
    for batch in _existing_batches(output_dir):
        if pq.read_schema(batch) != OUTPUT_SCHEMA:
            raise RuntimeError(
                f"Batch schema does not match {CONTRACT_VERSION}: {batch}"
            )
        table = pq.read_table(batch, columns=["track_id"])
        for track_id in table["track_id"].to_pylist():
            if track_id in seen_tracks:
                raise RuntimeError(f"Duplicate track_id across batches: {track_id}")
            seen_tracks.add(track_id)
            relative_path = file_by_track_id.get(track_id)
            if relative_path is None:
                raise RuntimeError(
                    f"Output track_id is absent from data_root: {track_id}"
                )
            recovered.add(relative_path)

    missing_output = checkpoint - recovered
    if missing_output:
        raise RuntimeError(
            f"Checkpoint contains {len(missing_output)} files with no output row",
        )
    if recovered != checkpoint:
        save_checkpoint(checkpoint_path, recovered)
    return recovered


def worker(
    task: tuple[str, Path],
) -> tuple[str, str | None, np.ndarray | None, str | None]:
    relative_path, path = task
    try:
        track_id, features = process_one_file(path)
        return relative_path, track_id, features, None
    except (KeyError, OSError, ValueError) as error:
        return relative_path, None, None, f"{type(error).__name__}: {error}"


def flush_batch(
    output_dir: Path,
    batch_index: int,
    track_ids: list[str],
    features: list[np.ndarray],
) -> Path:
    matrix = np.vstack(features)
    if matrix.shape != (len(track_ids), FEATURE_COUNT):
        raise RuntimeError(f"Invalid batch matrix shape: {matrix.shape}")
    if np.any(np.isinf(matrix)):
        raise RuntimeError("Batch contains Inf")
    arrays: list[pa.Array] = [pa.array(track_ids, type=pa.string())]
    arrays.extend(
        pa.array(
            matrix[:, index],
            mask=np.isnan(matrix[:, index]),
            type=pa.float64(),
        )
        for index in range(FEATURE_COUNT)
    )
    table = pa.Table.from_arrays(arrays, schema=OUTPUT_SCHEMA)
    output_path = output_dir / f"features_{batch_index:04d}.parquet"
    temporary = output_dir / f".features_{batch_index:04d}.parquet.tmp"
    pq.write_table(table, temporary, compression="snappy")
    temporary.replace(output_path)
    return output_path


def _next_batch_index(output_dir: Path) -> int:
    indices = [
        int(path.stem.rsplit("_", 1)[1]) for path in _existing_batches(output_dir)
    ]
    return max(indices, default=-1) + 1


def run_extraction(
    data_root: Path,
    output_dir: Path,
    *,
    workers: int = 1,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
) -> ExtractionResult:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_contract(data_root, output_dir)
    all_files = discover_files(data_root)
    relative_paths = [path.relative_to(data_root).as_posix() for path in all_files]
    file_by_track_id: dict[str, str] = {}
    for path, relative_path in zip(all_files, relative_paths, strict=True):
        if path.stem in file_by_track_id:
            raise RuntimeError(f"Duplicate HDF5 filename/track_id: {path.stem}")
        file_by_track_id[path.stem] = relative_path

    completed = recover_completed(output_dir, file_by_track_id)
    remaining = [
        (relative_path, path)
        for path, relative_path in zip(all_files, relative_paths, strict=True)
        if relative_path not in completed
    ]
    if limit is not None:
        remaining = remaining[:limit]

    print(
        f"Found {len(all_files)} .h5 files, {len(completed)} complete, "
        f"processing {len(remaining)} with {workers} worker(s)",
    )
    if not remaining:
        return ExtractionResult(len(all_files), len(completed), 0, 0, 0)

    checkpoint_path = output_dir / "checkpoint.txt"
    error_path = output_dir / "errors.txt"
    error_path.unlink(missing_ok=True)
    batch_index = _next_batch_index(output_dir)
    relative_batch: list[str] = []
    track_id_batch: list[str] = []
    feature_batch: list[np.ndarray] = []
    processed = 0
    failed = 0
    batches_written = 0

    def commit_batch() -> None:
        nonlocal batch_index, batches_written
        if not feature_batch:
            return
        output_path = flush_batch(
            output_dir, batch_index, track_id_batch, feature_batch
        )
        completed.update(relative_batch)
        save_checkpoint(checkpoint_path, completed)
        print(f"\nWrote {output_path.name}: {len(feature_batch)} rows")
        batch_index += 1
        batches_written += 1
        relative_batch.clear()
        track_id_batch.clear()
        feature_batch.clear()

    def consume(
        results: Iterable[tuple[str, str | None, np.ndarray | None, str | None]],
    ) -> None:
        nonlocal processed, failed
        for relative_path, track_id, features, error in results:
            processed += 1
            if error is not None or track_id is None or features is None:
                failed += 1
                with error_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"{relative_path}\t{error or 'unknown error'}\n")
                print(f"\nERROR {relative_path}: {error}", file=sys.stderr)
                continue
            expected_track_id = Path(relative_path).stem
            if track_id != expected_track_id:
                raise RuntimeError(
                    f"HDF5 track_id {track_id} does not match filename {expected_track_id}",
                )
            relative_batch.append(relative_path)
            track_id_batch.append(track_id)
            feature_batch.append(features)
            current = len(completed) + len(feature_batch)
            if current % 100 == 0 or processed == len(remaining):
                sys.stdout.write(f"\r[{current}/{len(all_files)}]")
                sys.stdout.flush()
            if len(feature_batch) >= batch_size:
                commit_batch()

    if workers > 1:
        with Pool(workers) as pool:
            consume(pool.imap(worker, remaining, chunksize=1))
    else:
        consume(worker(task) for task in remaining)
    commit_batch()

    if limit is None and failed == 0 and len(completed) != len(all_files):
        raise RuntimeError(
            f"Incomplete extraction: {len(completed)} of {len(all_files)} files",
        )
    print(
        f"Done. completed={len(completed)} processed={processed} failed={failed}",
    )
    return ExtractionResult(
        len(all_files),
        len(completed),
        processed,
        failed,
        batches_written,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Root of the MSD HDF5 tree")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many remaining files in this invocation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_extraction(
        args.data_root,
        args.output_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    if result.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
