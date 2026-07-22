
def _select_engine(requested: str) -> str:
    configured = os.environ.get("MERLIN_PARQUET_ENGINE", requested)
    if configured not in {"auto", "duckdb", "pyarrow"}:
        raise ValueError("MERLIN_PARQUET_ENGINE must be auto, duckdb, or pyarrow")
    if configured == "auto":
        return "duckdb" if importlib.util.find_spec("duckdb") else "pyarrow"
    return configured


def parquet_rows(
    path: str | Path,
    columns: Sequence[str],
    *,
    edge_type: str | None = None,
    order_by: Sequence[str] = (),
    engine: str = "auto",
) -> Iterable[tuple[object, ...]]:
    """Yield selected columns in bounded batches without importing both engines."""
    root = Path(path)
    selected = tuple(_validated_identifier(column) for column in columns)
    ordering = tuple(_validated_identifier(column) for column in order_by)
    if not selected:
        raise ValueError("at least one Parquet column is required")
    source = root / f"edge_type={edge_type}" if edge_type else root
    if not source.exists():
        raise FileNotFoundError(f"Parquet source does not exist: {source}")
    selected_engine = _select_engine(engine)
    if selected_engine == "duckdb":
        yield from _duckdb_rows(source, selected, ordering)
    else:
        yield from _pyarrow_rows(source, selected, ordering)


def _duckdb_rows(
    source: Path,
    columns: Sequence[str],
    order_by: Sequence[str],
) -> Iterable[tuple[object, ...]]:
    try:
        import duckdb
    except ImportError as error:
        raise RuntimeError("DuckDB Parquet loading requires duckdb") from error
    parquet_source = source if source.is_file() else source / "*.parquet"
    escaped = str(parquet_source).replace("'", "''")
    projection = ", ".join(columns)
    ordering = f" ORDER BY {', '.join(order_by)}" if order_by else ""
    relation = duckdb.sql(
        f"SELECT {projection} FROM read_parquet('{escaped}', hive_partitioning=true)"
        f"{ordering}"
    )
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
