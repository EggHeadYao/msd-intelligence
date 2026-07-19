# ruff: noqa: T201
"""Extract selected scalar fields from the MSD summary HDF5 file."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Sequence

EXPECTED_ROWS = 1_000_000

ANALYSIS_FIELDS: tuple[str, ...] = (
    "track_id",
    "loudness",
    "tempo",
    "duration",
    "key",
    "key_confidence",
    "mode",
    "mode_confidence",
    "time_signature",
    "time_signature_confidence",
    "end_of_fade_in",
    "start_of_fade_out",
)

METADATA_FIELDS: tuple[str, ...] = (
    "artist_id",
    "artist_name",
    "release",
    "release_7digitalid",
    "song_id",
    "song_hotttnesss",
    "artist_hotttnesss",
    "artist_familiarity",
    "title",
    "track_7digitalid",
)

STRING_FIELDS = frozenset(
    {"track_id", "artist_id", "artist_name", "release", "song_id", "title"},
)

POSITIVE_OR_NULL_FIELDS = frozenset(
    {
        "duration",
        "tempo",
        "time_signature",
        "release_7digitalid",
        "track_7digitalid",
        "year",
    },
)

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("track_id", pa.string(), nullable=False),
        pa.field("loudness", pa.float64()),
        pa.field("tempo", pa.float64()),
        pa.field("duration", pa.float64()),
        pa.field("key", pa.int32()),
        pa.field("key_confidence", pa.float64()),
        pa.field("mode", pa.int32()),
        pa.field("mode_confidence", pa.float64()),
        pa.field("time_signature", pa.int32()),
        pa.field("time_signature_confidence", pa.float64()),
        pa.field("end_of_fade_in", pa.float64()),
        pa.field("start_of_fade_out", pa.float64()),
        pa.field("artist_id", pa.string(), nullable=False),
        pa.field("artist_name", pa.string()),
        pa.field("release", pa.string()),
        pa.field("release_7digitalid", pa.int64()),
        pa.field("song_id", pa.string(), nullable=False),
        pa.field("song_hotttnesss", pa.float64()),
        pa.field("artist_hotttnesss", pa.float64()),
        pa.field("artist_familiarity", pa.float64()),
        pa.field("title", pa.string()),
        pa.field("track_7digitalid", pa.int64()),
        pa.field("year", pa.int32()),
    ],
)


def _decode(value: object) -> str:
    """Decode an HDF5 byte string."""
    if isinstance(value, bytes | bytearray | np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _clean_value(field: str, value: object) -> object | None:
    """Convert MSD missing-value sentinels to Arrow nulls."""
    if field in STRING_FIELDS:
        return _decode(value)

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if field in POSITIVE_OR_NULL_FIELDS and value <= 0:  # type: ignore[operator]
        return None
    return value


def _row_to_dict(row: np.void, fields: Sequence[str]) -> dict[str, object | None]:
    """Convert one compound HDF5 row into a plain dictionary."""
    return {field: _clean_value(field, row[field]) for field in fields}


def read_summary(h5: h5py.File, limit: int | None = None) -> pa.Table:
    """Read the selected summary fields and combine datasets by row position."""
    selection = slice(None) if limit is None else slice(0, limit)
    analysis_rows: np.ndarray = h5["/analysis/songs"][selection]
    metadata_rows: np.ndarray = h5["/metadata/songs"][selection]
    musicbrainz_rows: np.ndarray = h5["/musicbrainz/songs"][selection]

    row_count = analysis_rows.shape[0]
    if metadata_rows.shape[0] != row_count or musicbrainz_rows.shape[0] != row_count:
        raise RuntimeError(
            "Row count mismatch: "
            f"analysis={row_count}, metadata={metadata_rows.shape[0]}, "
            f"musicbrainz={musicbrainz_rows.shape[0]}",
        )

    rows: list[dict[str, object | None]] = []
    for index in range(row_count):
        row = _row_to_dict(analysis_rows[index], ANALYSIS_FIELDS)
        row.update(_row_to_dict(metadata_rows[index], METADATA_FIELDS))
        row["year"] = _clean_value("year", musicbrainz_rows[index]["year"])
        rows.append(row)

    return pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)


def main() -> None:
    """Read the full summary HDF5 file and write one Parquet file."""
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python -m tools.hdf5.extract_summary INPUT_H5 OUTPUT_PARQUET",
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    with h5py.File(input_path, "r") as h5:
        result = read_summary(h5)

    if result.num_rows != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, got {result.num_rows}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(result, output_path)
    print(
        f"Wrote {result.num_rows} rows, {result.num_columns} columns to {output_path}",
    )


if __name__ == "__main__":
    main()
