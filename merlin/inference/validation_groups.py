        "query_groups": list(VALIDATION_QUERY_GROUPS),
        "group_stats": {name: dict(group_stats[name]) for name in VALIDATION_QUERY_GROUPS},
        "thresholds_file": Path(thresholds_path).name,
        "thresholds_sha256": sha256_path(thresholds_path),
        "positives_path": Path(positives_path).name,
        "positives_sha256": sha256_path(positives_path),
        "validation_pairs_path": Path(validation_pairs_path).name,
        "validation_pairs_sha256": sha256_path(validation_pairs_path),
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_validation_group_manifest(
    manifest_path: str | Path,
    *,
    thresholds_path: str | Path,
    positives_path: str | Path,
    validation_pairs_path: str | Path,
    expected_scope: str,
    expected_parent_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != "set_b_validation_groups":
        raise ValueError("validation-group artifact type mismatch")
    if manifest.get("artifact_version") != VALIDATION_GROUP_VERSION:
        raise ValueError("validation-group artifact version mismatch")
    if manifest.get("scope") != expected_scope:
        raise ValueError("validation-group scope mismatch")
    if manifest.get("fit_split") != "set_a" or manifest.get("apply_split") != "set_b":
        raise ValueError("validation-group split boundary mismatch")
    if manifest.get("query_groups") != list(VALIDATION_QUERY_GROUPS):
        raise ValueError("validation-group names or order mismatch")
    with Path(thresholds_path).open("r", encoding="utf-8") as stream:
        thresholds = json.load(stream)
    if thresholds.get("artifact_type") != "set_b_validation_thresholds":
        raise ValueError("validation-group threshold artifact type mismatch")
    if thresholds.get("artifact_version") != VALIDATION_GROUP_VERSION:
        raise ValueError("validation-group threshold artifact version mismatch")
    if thresholds.get("fit_split") != "set_a":
        raise ValueError("validation-group threshold fit split mismatch")
    acoustic_p50 = float(thresholds.get("pre_pca_acoustic_cosine_p50", math.nan))
    acoustic_p90 = float(thresholds.get("pre_pca_acoustic_cosine_p90", math.nan))
    if not math.isfinite(acoustic_p50) or not math.isfinite(acoustic_p90):
        raise ValueError("validation-group acoustic thresholds are not finite")
    if acoustic_p50 > acoustic_p90:
        raise ValueError("validation-group acoustic thresholds are reversed")
    tag_threshold = float(thresholds.get("tag_positive_threshold", math.nan))
    if not math.isfinite(tag_threshold):
        raise ValueError("validation-group tag threshold is not finite")
    artifacts = (
        ("thresholds", Path(thresholds_path)),
        ("positives", Path(positives_path)),
        ("validation_pairs", Path(validation_pairs_path)),
    )
    for name, path in artifacts:
        path_key = "thresholds_file" if name == "thresholds" else f"{name}_path"
        if manifest.get(path_key) != path.name:
            raise ValueError(f"validation-group {name} path mismatch")
        if manifest.get(f"{name}_sha256") != sha256_path(path):
            raise ValueError(f"validation-group {name} hash mismatch")
    parents = manifest.get("parent_hashes")
    if not isinstance(parents, dict):
        raise ValueError("validation-group parent hashes are missing")
    for name, expected_hash in (expected_parent_hashes or {}).items():
        if parents.get(name) != expected_hash:
            raise ValueError(f"validation-group parent hash mismatch: {name}")
    stats = manifest.get("group_stats")
    if not isinstance(stats, dict) or set(stats) != set(VALIDATION_QUERY_GROUPS):
        raise ValueError("validation-group statistics are incomplete")
    return manifest
