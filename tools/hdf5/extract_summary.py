# ruff: noqa: T201
"""Extract scalar fields from msd_summary_file.h5 into a Parquet file."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np  # noqa: TC002  # used at runtime for ndarray iteration
import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_ROWS: int = 1000000

ANALYSIS_FIELDS: tuple[str, ...] = (
    "track_id",
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "duration",
    "key",
    "mode",
    "time_signature",
)

METADATA_FIELDS: tuple[str, ...] = (
    "artist_id",
    "artist_name",
    "release",
    "song_id",
    "song_hotttnesss",
    "artist_hotttnesss",
    "artist_familiarity",
    "title",
)

STRING_FIELDS: frozenset[str] = frozenset(
    {"track_id", "artist_id", "artist_name", "release", "song_id", "title"},
)


def _decode(val: object) -> str:
    """Decode HDF5 byte strings to Python str, pass through native str."""
    if isinstance(val, bytes | bytearray):
        return bytes(val).decode("utf-8")
    return str(val)


def _row_to_dict(row: np.void, fields: tuple[str, ...]) -> dict[str, object]:
    """Convert one HDF5 compound row into a plain dict, decoding bytes."""
    d: dict[str, object] = {}
    for f in fields:
        v = row[f]
        d[f] = _decode(v) if f in STRING_FIELDS else v
    return d


def read_summary(h5: h5py.File) -> pa.Table:
    """Read all scalar fields, combining the three datasets by position.

    Args:
        h5: Opened msd_summary_file.h5 handle.

    Returns:
        PyArrow Table with 1M rows and all scalar columns.

    """
    analysis_rows: np.ndarray = h5["/analysis/songs"][:]
    metadata_rows: np.ndarray = h5["/metadata/songs"][:]
    mb_rows: np.ndarray = h5["/musicbrainz/songs"][:]

    n: int = analysis_rows.shape[0]
    if metadata_rows.shape[0] != n or mb_rows.shape[0] != n:
        msg: str = (
            f"Row count mismatch: analysis={n}, "
            f"metadata={metadata_rows.shape[0]}, "
            f"musicbrainz={mb_rows.shape[0]}"
        )
        raise RuntimeError(msg)

    rows: list[dict[str, object]] = []
    for i in range(analysis_rows.shape[0]):
        d: dict[str, object] = _row_to_dict(analysis_rows[i], ANALYSIS_FIELDS)
        d.update(_row_to_dict(metadata_rows[i], METADATA_FIELDS))
        d["year"] = int(mb_rows[i]["year"])
        rows.append(d)

    return pa.Table.from_pylist(rows)


def main() -> None:
    """Entry point: read msd_summary_file.h5 and write songs_scalar_raw.parquet."""
    input_path: Path = Path(sys.argv[1])
    output_path: Path = Path(sys.argv[2])

    with h5py.File(input_path, "r") as h5:
        result: pa.Table = read_summary(h5)

    if result.num_rows != EXPECTED_ROWS:
        err: str = f"Expected {EXPECTED_ROWS} rows, got {result.num_rows}"
        raise RuntimeError(err)

    pq.write_table(result, str(output_path))
    print(
        f"Wrote {result.num_rows} rows, {result.num_columns} columns to {output_path}",
    )


if __name__ == "__main__":
    main()
