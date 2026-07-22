            pa.field("query_track_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("positives", pa.list_(pa.struct((
                pa.field("track_id", pa.string(), nullable=False),
                pa.field("positive_sources", pa.list_(pa.string()), nullable=False),
            ))), nullable=False),
        ))
    write_row_artifact(counted(), output, parquet_schema=parquet_schema)
    manifest = {
        "artifact_type": "weak_positives",
        "artifact_version": WEAK_LABEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "query_count": query_count,
        "positive_count": positive_count,
        "max_positives_per_query": MAX_POSITIVES_PER_QUERY,
        "source_counts": dict(sorted(source_counts.items())),
        "positives_file": output.name,
        "storage_format": "parquet" if output.suffix == ".parquet" else "jsonl_gzip",
        "positives_sha256": sha256_path(output),
        "thresholds_file": Path(thresholds_path).name,
        "thresholds_sha256": sha256_path(thresholds_path),
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_weak_positives(path: str | Path) -> dict[str, dict[str, frozenset[str]]]:
    result: dict[str, dict[str, frozenset[str]]] = {}
    for row in read_row_artifact(path):
        query_id = str(row["query_track_id"])
        if query_id in result:
            raise ValueError("weak positives contain a duplicate query")
        positives: dict[str, frozenset[str]] = {}
        for positive in row["positives"]:
            track_id = str(positive["track_id"])
            sources = frozenset(str(value) for value in positive["positive_sources"])
            if not sources or not sources.issubset(POSITIVE_SOURCES):
                raise ValueError("weak positive contains invalid provenance")
            if track_id in positives:
                raise ValueError("weak positive record contains a duplicate track")
            positives[track_id] = sources
        result[query_id] = positives
    return result


def load_weak_positive_manifest(
    manifest_path: str | Path,
    positives_path: str | Path,
    thresholds_path: str | Path,
    *,
    expected_scope: str,
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != "weak_positives":
        raise ValueError("weak-positive artifact type mismatch")
    if manifest.get("artifact_version") != WEAK_LABEL_VERSION:
        raise ValueError("weak-positive artifact version mismatch")
    if manifest.get("scope") != expected_scope:
        raise ValueError("weak-positive scope mismatch")
    positives = Path(positives_path)
    thresholds = Path(thresholds_path)
    if manifest.get("positives_file") != positives.name:
        raise ValueError("weak-positive output path mismatch")
    if manifest.get("positives_sha256") != sha256_path(positives):
        raise ValueError("weak-positive output hash mismatch")
    if manifest.get("thresholds_file") != thresholds.name:
        raise ValueError("weak-positive threshold path mismatch")
    if manifest.get("thresholds_sha256") != sha256_path(thresholds):
        raise ValueError("weak-positive threshold hash mismatch")
    return manifest
