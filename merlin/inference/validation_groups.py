        and not same_artist
        and has_release_pair
        and not same_release
        and tag is not None
        and tag < tag_positive_threshold
    )
    relation_signal = (
        same_artist
        or same_release
        or directed_artist_similarity
        or (tag is not None and tag >= tag_positive_threshold)
    )
    relation_dominant = acoustic < acoustic_p50 and relation_signal
    return audio_dominant, relation_dominant


def mixed_positive_ids(
    query_track_id: str,
    audio_positive_ids: set[str],
    relation_positive_ids: set[str],
    *,
    seed: int = VALIDATION_GROUP_SEED,
) -> tuple[str, ...]:
    """Select equal Audio/Relation counts without backfilling a short side."""
    import hashlib

    if not query_track_id:
        raise ValueError("mixed validation query ID must not be empty")
    overlap = audio_positive_ids & relation_positive_ids
    if overlap:
        raise ValueError("Audio- and Relation-dominant positives must be disjoint")
    count = min(len(audio_positive_ids), len(relation_positive_ids))

    def ordered(values: set[str], source: str) -> list[str]:
        return sorted(
            values,
            key=lambda candidate_id: (
                hashlib.sha256(
                    f"{seed}\0{query_track_id}\0{source}\0{candidate_id}".encode("utf-8")
                ).hexdigest(),
                candidate_id,
            ),
        )

    selected = ordered(audio_positive_ids, "audio")[:count]
    selected.extend(ordered(relation_positive_ids, "relation")[:count])
    return tuple(selected)


def write_validation_group_manifest(
    manifest_path: str | Path,
    *,
    thresholds_path: str | Path,
    positives_path: str | Path,
    validation_pairs_path: str | Path,
    parent_paths: Mapping[str, str | Path],
    scope: str,
    threshold_sample_count: int,
    group_stats: Mapping[str, Mapping[str, int | float]],
) -> dict[str, object]:
    if scope not in {"formal", "smoke"}:
        raise ValueError("validation-group scope must be formal or smoke")
    if set(group_stats) != set(VALIDATION_QUERY_GROUPS):
        raise ValueError("validation-group statistics must cover all frozen groups")
    if threshold_sample_count <= 0:
        raise ValueError("validation-group threshold sample must not be empty")
    manifest = {
        "artifact_type": "set_b_validation_groups",
        "artifact_version": VALIDATION_GROUP_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "fit_split": "set_a",
        "apply_split": "set_b",
        "seed": VALIDATION_GROUP_SEED,
        "threshold_sample_count": int(threshold_sample_count),
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
