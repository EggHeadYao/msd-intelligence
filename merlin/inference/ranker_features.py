                    **raw,
                }

    output = Path(output_path)
    parquet_schema = None
    if output.suffix == ".parquet":
        import pyarrow as pa

        fields = [
            pa.field("query_track_id", pa.string(), nullable=False),
            pa.field("candidate_track_id", pa.string(), nullable=False),
            pa.field("label", pa.int64(), nullable=False),
            pa.field("positive_sources", pa.list_(pa.string()), nullable=False),
            pa.field("negative_source", pa.string()),
            pa.field("recall_sources", pa.list_(pa.string()), nullable=False),
            pa.field("query_group", pa.string()),
            pa.field("eligible_positive_count", pa.int64()),
        ]
        fields.extend(pa.field(name, pa.float64()) for name in RAW_BASE_FEATURES)
        parquet_schema = pa.schema(fields)
    row_count = write_row_artifact(rows(), output, parquet_schema=parquet_schema)
    if row_count == 0:
        raise ValueError("raw feature artifact must not be empty")
    manifest = {
        "artifact_type": "ranker_raw_pair_features",
        "artifact_version": RAW_FEATURE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "pair_kind": pair_kind,
        "stage": stage,
        "feature_schema_version": RANKER_V2_SCHEMA_VERSION,
        "raw_feature_order": list(RAW_BASE_FEATURES),
        "row_count": row_count,
        "counts": dict(sorted(counts.items())),
        "output_file": output.name,
        "storage_format": "parquet" if output.suffix == ".parquet" else "jsonl_gzip",
        "output_sha256": sha256_path(output),
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_raw_feature_manifest(
    manifest_path: str | Path,
    feature_path: str | Path,
    *,
    expected_scope: str,
    expected_pair_kind: str | None = None,
    expected_stage: str | None = None,
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != "ranker_raw_pair_features":
        raise ValueError("raw-feature artifact type mismatch")
    if manifest.get("artifact_version") != RAW_FEATURE_VERSION:
        raise ValueError("raw-feature artifact version mismatch")
    if manifest.get("feature_schema_version") != RANKER_V2_SCHEMA_VERSION:
        raise ValueError("raw-feature schema version mismatch")
    if manifest.get("raw_feature_order") != list(RAW_BASE_FEATURES):
        raise ValueError("raw-feature order mismatch")
    if manifest.get("scope") != expected_scope:
        raise ValueError("raw-feature scope mismatch")
    if expected_pair_kind is not None and manifest.get("pair_kind") != expected_pair_kind:
        raise ValueError("raw-feature pair kind mismatch")
    if expected_stage is not None and manifest.get("stage") != expected_stage:
        raise ValueError("raw-feature stage mismatch")
    features = Path(feature_path)
    if manifest.get("output_file") != features.name:
        raise ValueError("raw-feature output path mismatch")
    if manifest.get("output_sha256") != sha256_path(features):
        raise ValueError("raw-feature output hash mismatch")
    return manifest
