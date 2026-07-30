from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

BATCH_SIZE = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MSD metadata to Parquet")
    parser.add_argument("--input", type=Path, default=Path("msd/AdditionalFiles"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parquets/year_prediction/raw/metadata"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def export_query(
    database: Path,
    query: str,
    output: Path,
    schema: pa.Schema,
) -> int:
    connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
    cursor = connection.execute(query)
    writer = pq.ParquetWriter(output, schema, compression="snappy")
    count = 0
    try:
        while rows := cursor.fetchmany(BATCH_SIZE):
            columns = list(zip(*rows, strict=True))
            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array(values, type=field.type)
                    for values, field in zip(columns, schema, strict=True)
                ],
                schema=schema,
            )
            writer.write_batch(batch)
            count += len(rows)
    finally:
        writer.close()
        connection.close()
    return count


def export_locations(source: Path, output: Path) -> int:
    schema = pa.schema(
        [
            ("artist_id", pa.string()),
            ("latitude", pa.float64()),
            ("longitude", pa.float64()),
            ("location", pa.string()),
        ]
    )
    rows: list[tuple[str, float, float, str]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            artist_id, latitude, longitude, _name, location = line.rstrip("\n").split(
                "<SEP>", 4
            )
            rows.append((artist_id, float(latitude), float(longitude), location))
    records = [dict(zip(schema.names, row, strict=True)) for row in rows]
    pq.write_table(pa.Table.from_pylist(records, schema), output)
    return len(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    term_db = args.input / "artist_term.db"
    similarity_db = args.input / "artist_similarity.db"
    tag_schema = pa.schema(
        [("artist_id", pa.string()), ("tag", pa.string()), ("source", pa.string())]
    )
    similarity_schema = pa.schema(
        [
            ("artist_id", pa.string()),
            ("similar_artist_id", pa.string()),
            ("edge_order", pa.int64()),
        ]
    )
    counts = {
        "artist_tags": export_query(
            term_db,
            "SELECT artist_id, term, 'term' FROM artist_term "
            "UNION ALL SELECT artist_id, mbtag, 'mbtag' FROM artist_mbtag",
            args.output / "artist_tags.parquet",
            tag_schema,
        ),
        "artist_similarity": export_query(
            similarity_db,
            "SELECT target, similar, rowid FROM similarity ORDER BY rowid",
            args.output / "artist_similarity.parquet",
            similarity_schema,
        ),
        "artist_location": export_locations(
            args.input / "artist_location.txt",
            args.output / "artist_location.parquet",
        ),
    }
    manifest = {
        "contract_version": "year_prediction_raw_metadata_v1",
        "counts": counts,
        "sources": {
            "artist_term.db": file_sha256(term_db),
            "artist_similarity.db": file_sha256(similarity_db),
            "artist_location.txt": file_sha256(args.input / "artist_location.txt"),
        },
    }
    write_json(args.output / "manifest.json", manifest)
    return manifest


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
