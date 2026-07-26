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

