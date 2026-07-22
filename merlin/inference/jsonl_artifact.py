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


def read_row_artifact(path: str | Path) -> Iterator[dict[str, object]]:
    """Read either a Parquet row artifact or the historical gzip JSONL form."""
    source = Path(path)
    if source.suffix != ".parquet":
        yield from read_jsonl_gzip(source)
        return
    if not source.is_file():
        raise FileNotFoundError(f"Parquet row artifact does not exist: {source}")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("reading Parquet row artifacts requires pyarrow") from error
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=100_000):
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
