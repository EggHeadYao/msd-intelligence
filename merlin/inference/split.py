            }

    output = Path(assignments_path)
    parquet_schema = None
    if output.suffix == ".parquet":
        import pyarrow as pa

        parquet_schema = pa.schema((
            pa.field("track_id", pa.string(), nullable=False),
            pa.field("split_key", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
        ))
    row_count = write_row_artifact(assignments(), output, parquet_schema=parquet_schema)
    if row_count == 0:
        raise ValueError("split input must not be empty")
    manifest = {
        "artifact_type": "supervised_split",
        "artifact_version": SPLIT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "seed": SPLIT_SEED,
        "split_key": "valid song_id else track_id",
        "thresholds": {"set_a": 0.10, "set_b": 0.01, "set_c": 0.02},
        "track_count": row_count,
        "group_count": len(group_splits),
        "track_counts": dict(sorted(counts.items())),
        "assignments_file": output.name,
        "storage_format": "parquet" if output.suffix == ".parquet" else "jsonl_gzip",
        "assignments_sha256": sha256_path(output),
        "songs_metadata_sha256": sha256_path(songs_metadata_path),
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_split_manifest(
    manifest_path: str | Path,
    assignments_path: str | Path,
    *,
    expected_scope: str | None = None,
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != "supervised_split":
        raise ValueError("split artifact type mismatch")
    if manifest.get("artifact_version") != SPLIT_VERSION:
        raise ValueError("split artifact version mismatch")
    assignments = Path(assignments_path)
    if manifest.get("assignments_file") != assignments.name:
        raise ValueError("split assignment path mismatch")
    if manifest.get("assignments_sha256") != sha256_path(assignments):
        raise ValueError("split assignment hash mismatch")
    if expected_scope is not None and manifest.get("scope") != expected_scope:
        raise ValueError("split scope mismatch")
    return manifest


def load_split_assignments(path: str | Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    group_assignments: dict[str, str] = {}
    for row in read_row_artifact(path):
        track_id = str(row["track_id"])
        key = str(row["split_key"])
        split = str(row["split"])
        if split not in {"set_a", "set_b", "set_c", "remaining"}:
            raise ValueError("split assignment contains an unknown set")
        if track_id in assignments:
            raise ValueError("split assignments contain a duplicate track")
        if key in group_assignments and group_assignments[key] != split:
            raise ValueError("one split group crosses multiple sets")
        assignments[track_id] = split
        group_assignments[key] = split
    if not assignments:
        raise ValueError("split assignments must not be empty")
    return assignments
