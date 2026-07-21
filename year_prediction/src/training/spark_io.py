from __future__ import annotations

import sys

from pyspark import cloudpickle

cloudpickle.register_pickle_by_value(sys.modules[__name__])

import os
import shutil
from pathlib import Path
from typing import Iterator, Sequence

from pyspark import TaskContext
from pyspark.sql import DataFrame, Row


def _write_partition(
    index: int,
    rows: Iterator[Row],
    output: str,
    columns: tuple[str, ...],
    batch_size: int,
) -> Iterator[int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    context = TaskContext.get()
    attempt = 0 if context is None else context.attemptNumber()
    final_path = os.path.join(output, f"part-{index:05d}.parquet")
    temporary = final_path + f".attempt-{attempt}.tmp"
    writer = None
    batch: list[dict[str, object]] = []
    count = 0

    def flush() -> None:
        nonlocal writer, batch
        if not batch:
            return
        table = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
        writer.write_table(table)
        batch = []

    try:
        for row in rows:
            batch.append({name: row[name] for name in columns})
            count += 1
            if len(batch) >= batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    if count:
        os.replace(temporary, final_path)
    elif os.path.exists(temporary):
        os.remove(temporary)
    yield count


def write_parquet_parts(
    frame: DataFrame,
    output: Path,
    columns: Sequence[str] | None = None,
    batch_size: int = 4096,
) -> int:
    if output.exists():
        raise FileExistsError(f"Parquet output already exists: {output}")
    output.mkdir(parents=True)
    selected = tuple(columns or frame.columns)
    counts = frame.select(*selected).rdd.mapPartitionsWithIndex(
        lambda index, rows: _write_partition(
            index, rows, str(output), selected, batch_size
        )
    ).collect()
    total = int(sum(counts))
    if total <= 0:
        shutil.rmtree(output)
        raise ValueError("cannot write an empty Parquet dataset")
    (output / "_SUCCESS").write_text("", encoding="ascii")
    return total


def write_native_model(path: Path, model_text: str) -> None:
    if path.exists():
        raise FileExistsError(f"model output already exists: {path}")
    path.write_text(model_text, encoding="ascii", newline="\n")
