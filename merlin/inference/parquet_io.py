    while rows := relation.fetchmany(100_000):
        yield from rows


def _pyarrow_rows(
    source: Path,
    columns: Sequence[str],
    order_by: Sequence[str],
) -> Iterable[tuple[object, ...]]:
    try:
        import pyarrow.dataset as ds
    except ImportError as error:
        raise RuntimeError("PyArrow Parquet loading requires pyarrow") from error
    dataset = ds.dataset(str(source), format="parquet", partitioning="hive")
    rows: Iterable[tuple[object, ...]] = (
        tuple(values)
        for batch in dataset.to_batches(columns=list(columns))
        for values in zip(*(batch.column(index).to_pylist() for index in range(len(columns))))
    )
    if order_by:
        indexes = tuple(columns.index(column) for column in order_by)
        yield from sorted(rows, key=lambda row: tuple(row[index] for index in indexes))
    else:
        yield from rows
