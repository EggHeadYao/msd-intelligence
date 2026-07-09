from __future__ import annotations

import h5py  # noqa: TC002
import pyarrow as pa


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
