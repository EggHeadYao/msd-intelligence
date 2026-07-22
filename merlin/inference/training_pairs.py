    output = Path(output_path)
    parquet_schema = None
    if output.suffix == ".parquet":
        import pyarrow as pa

        parquet_schema = pa.schema((
            pa.field("query_track_id", pa.string(), nullable=False),
            pa.field("candidate_track_id", pa.string(), nullable=False),
            pa.field("label", pa.int64(), nullable=False),
            pa.field("positive_sources", pa.list_(pa.string()), nullable=False),
            pa.field("negative_source", pa.string()),
            pa.field("recall_sources", pa.list_(pa.string()), nullable=False),
        ))
    pair_count = write_row_artifact(pair_rows(), output, parquet_schema=parquet_schema)
    if query_count == 0 or pair_count == 0:
        raise ValueError("training-pair artifact has no eligible query pairs")
    if totals["negative_count"] != NEGATIVE_RATIO * totals["positive_count"]:
        raise ValueError("training-pair artifact violates the 1:3 ratio")
    manifest = {
        "artifact_type": "ranker_training_pairs",
        "artifact_version": TRAINING_PAIR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "stage": stage,
        "seed": PAIR_SEED,
        "negative_ratio": NEGATIVE_RATIO,
        "candidate_aware_target_fraction": CANDIDATE_AWARE_FRACTION,
        "query_count": query_count,
        "pair_count": pair_count,
        "counts": dict(sorted(totals.items())),
        "actual_candidate_aware_fraction": (
            totals["candidate_aware_count"] / totals["negative_count"]
        ),
        "rejection_counts": dict(sorted(rejection_totals.items())),
        "pairs_file": output.name,
        "storage_format": "parquet" if output.suffix == ".parquet" else "jsonl_gzip",
        "pairs_sha256": sha256_path(output),
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_training_pair_manifest(
    manifest_path: str | Path,
    pairs_path: str | Path,
    *,
    expected_scope: str,
    expected_stage: str,
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != "ranker_training_pairs":
        raise ValueError("training-pair artifact type mismatch")
    if manifest.get("artifact_version") != TRAINING_PAIR_VERSION:
        raise ValueError("training-pair artifact version mismatch")
    if manifest.get("scope") != expected_scope:
        raise ValueError("training-pair scope mismatch")
    if manifest.get("stage") != expected_stage:
        raise ValueError("training-pair stage mismatch")
    pairs = Path(pairs_path)
    if manifest.get("pairs_file") != pairs.name:
        raise ValueError("training-pair output path mismatch")
    if manifest.get("pairs_sha256") != sha256_path(pairs):
        raise ValueError("training-pair output hash mismatch")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("training-pair counts are missing")
    if int(counts.get("negative_count", -1)) != NEGATIVE_RATIO * int(
        counts.get("positive_count", -1)
    ):
        raise ValueError("training-pair manifest violates the 1:3 ratio")
    return manifest
