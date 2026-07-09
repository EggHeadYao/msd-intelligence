# ruff: noqa: D100, D103, T201
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_ROWS: int = 1000000


def read_analysis_songs(h5: h5py.File) -> pa.Table:
    fields: tuple[str, ...] = (
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
    rows: list[dict[str, str | float | int]] = []
    ds: h5py.Dataset = h5["/analysis/songs"]
    for i in range(ds.shape[0]):
        row = ds[i]
        rows.append({f: row[f] for f in fields})
    return pa.Table.from_pylist(rows)


def read_metadata_songs(h5: h5py.File) -> pa.Table:
    fields: tuple[str, ...] = (
        "track_id",
        "artist_id",
        "artist_name",
        "release",
        "song_hotttnesss",
        "artist_hotttnesss",
        "artist_familiarity",
        "title",
    )
    rows: list[dict[str, str | float | int]] = []
    ds: h5py.Dataset = h5["/metadata/songs"]
    for i in range(ds.shape[0]):
        row = ds[i]
        rows.append({f: row[f] for f in fields})
    return pa.Table.from_pylist(rows)


def read_musicbrainz_songs(h5: h5py.File) -> pa.Table:
    rows: list[dict[str, int | str]] = []
    ds: h5py.Dataset = h5["/musicbrainz/songs"]
    for i in range(ds.shape[0]):
        row = ds[i]
        rows.append({"track_id": row["track_id"], "year": int(row["year"])})
    return pa.Table.from_pylist(rows)


def join_tables(
    analysis: pa.Table,
    metadata: pa.Table,
    musicbrainz: pa.Table,
) -> pa.Table:
    joined: pa.Table = analysis.join(metadata, keys="track_id")
    return joined.join(musicbrainz, keys="track_id")


def main() -> None:
    input_path: Path = Path(sys.argv[1])
    output_path: Path = Path(sys.argv[2])

    with h5py.File(input_path, "r") as h5:
        analysis: pa.Table = read_analysis_songs(h5)
        metadata: pa.Table = read_metadata_songs(h5)
        musicbrainz: pa.Table = read_musicbrainz_songs(h5)

    result: pa.Table = join_tables(analysis, metadata, musicbrainz)

    if result.num_rows != EXPECTED_ROWS:
        err: str = f"Expected {EXPECTED_ROWS} rows, got {result.num_rows}"
        raise RuntimeError(err)

    pq.write_table(result, str(output_path))
    print(
        f"Wrote {result.num_rows} rows, {result.num_columns} columns to {output_path}",
    )


if __name__ == "__main__":
    main()
