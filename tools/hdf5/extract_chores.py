# ruff: noqa: T201
"""Export supplementary MSD files (SQLite DBs, text tables) to Parquet."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def export_track_metadata(db_path: Path, output_dir: Path) -> None:
    """Export track_metadata.db -> track_metadata.parquet (1M rows)."""
    con: sqlite3.Connection = sqlite3.connect(
        f"file:{db_path}?mode=ro&immutable=1",
        uri=True,
    )
    con.row_factory = sqlite3.Row
    rows: list[dict[str, object]] = [
        dict(r) for r in con.execute("SELECT * FROM songs")
    ]
    con.close()
    _write(pa.Table.from_pylist(rows), output_dir / "track_metadata.parquet")


def export_artist_term(db_path: Path, output_dir: Path) -> None:
    """Export artist_term.db tables to Parquet."""
    con: sqlite3.Connection = sqlite3.connect(
        f"file:{db_path}?mode=ro&immutable=1",
        uri=True,
    )
    con.row_factory = sqlite3.Row

    rows: list[dict[str, object]] = [
        dict(r) for r in con.execute("SELECT * FROM artist_term")
    ]
    _write(pa.Table.from_pylist(rows), output_dir / "artist_term.parquet")

    rows2: list[dict[str, object]] = [
        dict(r) for r in con.execute("SELECT * FROM artist_mbtag")
    ]
    _write(pa.Table.from_pylist(rows2), output_dir / "artist_mbtag.parquet")
    con.close()


def export_sep_txt(path: Path, output_dir: Path, columns: tuple[str, ...]) -> None:
    """Export a <SEP>-delimited text file to Parquet."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            stripped: str = raw_line.rstrip("\n")
            if not stripped:
                continue
            parts: list[str] = stripped.split("<SEP>")
            rows.append(dict(zip(columns, parts, strict=False)))
    _write(pa.Table.from_pylist(rows), output_dir / f"{path.stem}.parquet")


def _write(table: pa.Table, path: Path) -> None:
    pq.write_table(table, str(path))
    print(f"  {path.name}: {table.num_rows} rows x {table.num_columns} cols")


def main() -> None:
    """Entry point: export all supplementary files to Parquet."""
    db_dir: Path = Path(sys.argv[1])  # AdditionalFiles/
    out_dir: Path = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    export_track_metadata(db_dir / "track_metadata.db", out_dir)
    export_artist_term(db_dir / "artist_term.db", out_dir)

    export_sep_txt(
        db_dir / "tracks_per_year.txt",
        out_dir,
        ("year", "track_id", "artist_name", "title"),
    )
    export_sep_txt(
        db_dir / "artist_location.txt",
        out_dir,
        ("artist_id", "latitude", "longitude", "name", "city"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
