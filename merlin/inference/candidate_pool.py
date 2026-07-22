            pa.field("query_track_id", pa.string(), nullable=False),
            pa.field("candidates", pa.list_(pa.struct((
                pa.field("track_id", pa.string(), nullable=False),
                pa.field("recall_sources", pa.list_(pa.string()), nullable=False),
                pa.field("recall_scores", pa.map_(pa.string(), pa.float64()), nullable=False),
                pa.field("source_ranks", pa.map_(pa.string(), pa.int64()), nullable=False),
            ))), nullable=False),
            pa.field("audit", pa.struct((
                pa.field("raw_candidates", pa.int64(), nullable=False),
                pa.field("unique_candidates", pa.int64(), nullable=False),
                pa.field("duplicate_candidates", pa.int64(), nullable=False),
                pa.field("source_available", pa.map_(pa.string(), pa.bool_()), nullable=False),
                pa.field("source_counts", pa.map_(pa.string(), pa.int64()), nullable=False),
                pa.field("source_shortages", pa.map_(pa.string(), pa.int64()), nullable=False),
            )), nullable=False),
        ))
    row_count = write_row_artifact(rows(), output, parquet_schema=parquet_schema)
    parents = {
        name: sha256_path(path) for name, path in sorted(parent_paths.items())
    }
    manifest = {
        "artifact_type": "candidate_pool",
        "artifact_version": CANDIDATE_POOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "query_count": row_count,
        "totals": totals,
        "source_totals": source_totals,
        "output_file": output.name,
        "storage_format": "parquet" if output.suffix == ".parquet" else "jsonl_gzip",
        "output_sha256": sha256_path(output),
        "parent_hashes": parents,
        "schema": {
            "query_track_id": "string",
            "candidates": "ordered array<track_id, recall_sources, recall_scores, source_ranks>",
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_candidate_pool_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    expected_scope: str | None = None,
    expected_parent_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != "candidate_pool":
        raise ValueError("candidate pool artifact type mismatch")
    if manifest.get("artifact_version") != CANDIDATE_POOL_VERSION:
        raise ValueError("candidate pool artifact version mismatch")
    if manifest.get("candidate_policy_version") != CANDIDATE_POLICY_VERSION:
        raise ValueError("candidate pool policy version mismatch")
    output = Path(output_path)
    if manifest.get("output_file") != output.name:
        raise ValueError("candidate pool output path mismatch")
    if manifest.get("output_sha256") != sha256_path(output):
        raise ValueError("candidate pool output hash mismatch")
    if expected_scope is not None and manifest.get("scope") != expected_scope:
        raise ValueError("candidate pool scope mismatch")
    parents = manifest.get("parent_hashes")
    if not isinstance(parents, dict):
        raise ValueError("candidate pool parent hashes are missing")
    for name, expected_hash in (expected_parent_hashes or {}).items():
        if parents.get(name) != expected_hash:
            raise ValueError(f"candidate pool parent hash mismatch: {name}")
    return manifest


def iter_candidate_pool(path: str | Path) -> Iterator[dict[str, object]]:
    yield from read_row_artifact(path)
