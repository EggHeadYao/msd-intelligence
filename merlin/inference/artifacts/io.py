"""Deterministic JSON, JSONL, and Parquet IO for C3 artifacts."""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Iterator, Mapping, Sequence


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def write_jsonl_gzip(
    rows: Iterable[Mapping[str, object]],
    path: str | Path,
) -> int:
    """Atomically write canonical JSON lines with a reproducible gzip header."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(dict(row), separators=(",", ":"), sort_keys=True))
                    stream.write("\n")
                    count += 1
    temporary.replace(output)
    return count


def read_jsonl_gzip(path: str | Path) -> Iterator[dict[str, object]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSONL artifact does not exist: {source}")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            yield row


def write_row_artifact(
    rows: Iterable[Mapping[str, object]],
    path: str | Path,
    *,
    parquet_schema: Any = None,
    batch_size: int = 10_000,
) -> int:
    """Write a streaming row artifact as Parquet or deterministic gzip JSONL."""
    output = Path(path)
    if output.suffix != ".parquet":
        return write_jsonl_gzip(rows, output)
    if parquet_schema is None:
        raise ValueError("Parquet row artifacts require an explicit schema")
    if batch_size <= 0:
        raise ValueError("Parquet batch size must be positive")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("writing Parquet row artifacts requires pyarrow") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    writer = pq.ParquetWriter(
        temporary,
        parquet_schema,
        compression="zstd",
        use_dictionary=True,
    )
    count = 0
    batch: list[Mapping[str, object]] = []
    try:
        for row in rows:
            batch.append(row)
            if len(batch) == batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=parquet_schema))
                count += len(batch)
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=parquet_schema))
            count += len(batch)
    finally:
        writer.close()
    if count == 0:
        temporary.unlink(missing_ok=True)
        return 0
    temporary.replace(output)
    return count


def _write_parquet_part(pa: Any, pq: Any, rows: list[Mapping[str, object]], schema: Any, path: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


class PartitionedParquetWriter:
    """Incrementally write an atomically published directory of Parquet parts."""

    def __init__(self, path: str | Path, schema: Any, *, rows_per_file: int = 250_000) -> None:
        if rows_per_file <= 0:
            raise ValueError("Parquet rows per file must be positive")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("writing Parquet datasets requires pyarrow") from error
        self._pa = pa
        self._pq = pq
        self.output = Path(path)
        if self.output.suffix != ".parquet":
            raise ValueError("partitioned Parquet output must end in .parquet")
        self.temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        if self.output.exists() or self.temporary.exists():
            raise FileExistsError(f"Parquet output or temporary path already exists: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.mkdir()
        self.schema = schema
        self.rows_per_file = rows_per_file
        self.count = 0
        self.part_count = 0
        self._buffer: list[Mapping[str, object]] = []
        self._closed = False

    def write_rows(self, rows: Iterable[Mapping[str, object]]) -> None:
        if self._closed:
            raise ValueError("cannot write to a closed Parquet dataset")
        for row in rows:
            self._buffer.append(row)
            if len(self._buffer) == self.rows_per_file:
                self._flush()

    @property
    def pending_count(self) -> int:
        return self.count + len(self._buffer)

    def _flush(self) -> None:
        if not self._buffer:
            return
        part = self.temporary / f"part-{self.part_count:05d}.parquet"
        _write_parquet_part(self._pa, self._pq, self._buffer, self.schema, part)
        self.count += len(self._buffer)
        self.part_count += 1
        self._buffer.clear()

    def close(self) -> int:
        if self._closed:
            return self.count
        self._flush()
        self._closed = True
        if self.count == 0:
            shutil.rmtree(self.temporary)
            raise ValueError("partitioned Parquet artifact must not be empty")
        (self.temporary / "_SUCCESS").write_bytes(b"")
        self.temporary.replace(self.output)
        return self.count

    def __enter__(self) -> PartitionedParquetWriter:
        return self

    def __exit__(self, error_type: object, _error: object, _traceback: object) -> None:
        if error_type is None:
            self.close()


def read_row_artifact(
    path: str | Path,
    *,
    batch_size: int = 100_000,
) -> Iterator[dict[str, object]]:
    """Read either a Parquet row artifact or the historical gzip JSONL form."""
    if batch_size <= 0:
        raise ValueError("row artifact read batch size must be positive")
    source = Path(path)
    if source.suffix != ".parquet":
        yield from read_jsonl_gzip(source)
        return
    if not source.exists():
        raise FileNotFoundError(f"Parquet row artifact does not exist: {source}")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("reading Parquet row artifacts requires pyarrow") from error
    files = (source,) if source.is_file() else tuple(sorted(source.glob("part-*.parquet")))
    if not files:
        raise ValueError(f"Parquet dataset contains no part files: {source}")
    for file_path in files:
        parquet = pq.ParquetFile(file_path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist(maps_as_pydicts="strict"):
                yield row


def write_json_atomic(payload: Mapping[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)
