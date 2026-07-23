"""Frozen Set-B Audio/Relation/Mixed validation-group contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ..artifact_lineage import artifact_size_bytes, sha256_path
from ..jsonl_artifact import write_json_atomic


VALIDATION_GROUP_VERSION = "merlin_validation_groups_v1"
VALIDATION_GROUP_SEED = 42
VALIDATION_QUERY_GROUPS = ("audio_dominant", "relation_dominant", "mixed")


def estimate_validation_scratch_gb(
    *,
    set_a_tracks: int,
    set_b_tracks: int,
    feature_dimension: int,
    unique_candidates: int,
    max_sample_pairs: int,
) -> float:
    """Estimate peak local Spark data for formal validation-group construction."""
    values = (
        set_a_tracks,
        set_b_tracks,
        feature_dimension,
        unique_candidates,
        max_sample_pairs,
    )
    if any(value <= 0 for value in values):
        raise ValueError("validation scratch estimate inputs must be positive")
    vector_checkpoint = (set_a_tracks + set_b_tracks) * feature_dimension * 8 * 1.5
    sample_sort = max_sample_pairs * 48 * 2
    validation_expansion = unique_candidates * len(VALIDATION_QUERY_GROUPS) * 64 * 2
    threshold_pairs = set_b_tracks * set_b_tracks * 0.15 * 48 * 2
    projected_bytes = (
        vector_checkpoint + sample_sort + validation_expansion + threshold_pairs
    )
    gib = projected_bytes / (1024**3)
    return max(4.0, math.ceil(gib * 4) / 4)


def write_audio_threshold_pairs_numpy(
    query_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    threshold: float,
    block_size: int = 256,
) -> int:
    """Write blocked exact cosine-threshold pairs without a Spark cross product."""
    if block_size <= 0 or not math.isfinite(threshold):
        raise ValueError("audio threshold and block size must be valid")
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("NumPy audio threshold search requires numpy and pyarrow") from error

    queries = sorted(query_rows, key=lambda row: str(row["query_track_id"]))
    candidates = sorted(candidate_rows, key=lambda row: str(row["candidate_track_id"]))
    if not queries or not candidates:
        raise ValueError("audio threshold search requires query and candidate vectors")

    def normalized_matrix(rows, vector_key: str, norm_key: str):
        matrix = np.asarray([row[vector_key] for row in rows], dtype=np.float64)
        norms = np.asarray([row[norm_key] for row in rows], dtype=np.float64)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("audio threshold vectors are invalid")
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("audio threshold vector norms are invalid")
        return matrix / norms[:, None]

    candidate_matrix = normalized_matrix(candidates, "c_vector", "c_norm")
    candidate_tracks = np.asarray(
        [str(row["candidate_track_id"]) for row in candidates], dtype=object
    )
    candidate_songs = np.asarray([row.get("c_song_id") for row in candidates], dtype=object)
    candidate_artists = np.asarray(
        [str(row["c_artist_id"]) for row in candidates], dtype=object
    )
    candidate_releases = np.asarray([row["c_release_id"] for row in candidates], dtype=object)

    schema = pa.schema((
        pa.field("query_track_id", pa.string(), nullable=False),
        pa.field("candidate_track_id", pa.string(), nullable=False),
        pa.field("q_artist_id", pa.string(), nullable=False),
        pa.field("c_artist_id", pa.string(), nullable=False),
    ))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, schema, compression="zstd", use_dictionary=True)
    count = 0
    try:
        for start in range(0, len(queries), block_size):
            block = queries[start : start + block_size]
            query_matrix = normalized_matrix(block, "q_vector", "q_norm")
            mask = query_matrix @ candidate_matrix.T >= threshold
            query_tracks = np.asarray(
                [str(row["query_track_id"]) for row in block], dtype=object
            )
            query_songs = np.asarray([row.get("q_song_id") for row in block], dtype=object)
            query_artists = np.asarray(
                [str(row["q_artist_id"]) for row in block], dtype=object
            )
            query_releases = np.asarray([row["q_release_id"] for row in block], dtype=object)
            mask &= query_tracks[:, None] != candidate_tracks[None, :]
            mask &= query_artists[:, None] != candidate_artists[None, :]
            mask &= query_releases[:, None] != candidate_releases[None, :]
            same_song = (
                (query_songs[:, None] != None)  # noqa: E711
                & (candidate_songs[None, :] != None)  # noqa: E711
                & (query_songs[:, None] == candidate_songs[None, :])
            )
            mask &= ~same_song
            query_positions, candidate_positions = np.nonzero(mask)
            if not len(query_positions):
                continue
            writer.write_table(pa.Table.from_arrays(
                (
                    pa.array(query_tracks[query_positions].tolist(), type=pa.string()),
                    pa.array(candidate_tracks[candidate_positions].tolist(), type=pa.string()),
                    pa.array(query_artists[query_positions].tolist(), type=pa.string()),
                    pa.array(candidate_artists[candidate_positions].tolist(), type=pa.string()),
                ),
                schema=schema,
            ))
            count += len(query_positions)
    finally:
        writer.close()
    temporary.replace(output)
    return count


def classify_validation_pair(
    pair: Mapping[str, object],
    *,
    acoustic_p50: float,
    acoustic_p90: float,
    tag_positive_threshold: float,
) -> tuple[bool, bool]:
    """Return Audio-dominant and Relation-dominant membership for one pair."""
    if not all(math.isfinite(value) for value in (
        acoustic_p50,
        acoustic_p90,
        tag_positive_threshold,
    )):
        raise ValueError("validation-group thresholds must be finite")
    if not acoustic_p50 <= acoustic_p90:
        raise ValueError("pre-PCA acoustic thresholds must satisfy p50 <= p90")
    acoustic = float(pair["pre_pca_cosine"])
    if not math.isfinite(acoustic):
        raise ValueError("pre-PCA acoustic cosine must be finite")
    tag_value = pair.get("tag_tfidf_cosine")
    tag = None if tag_value is None else float(tag_value)
    if tag is not None and not math.isfinite(tag):
        raise ValueError("artist-term cosine must be finite when present")

    same_song = bool(pair.get("same_song", False))
    same_artist = bool(pair.get("same_artist", False))
    same_release = bool(pair.get("same_release", False))
    has_artist_pair = bool(pair.get("has_artist_pair", False))
    has_release_pair = bool(pair.get("has_release_pair", False))
    directed_artist_similarity = bool(pair.get("directed_artist_similarity", False))
    if same_song:
        return False, False

    audio_dominant = (
        acoustic >= acoustic_p90
        and has_artist_pair
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
        "positives_size_bytes": artifact_size_bytes(positives_path),
        "validation_pairs_path": Path(validation_pairs_path).name,
        "validation_pairs_sha256": sha256_path(validation_pairs_path),
        "validation_pairs_size_bytes": artifact_size_bytes(validation_pairs_path),
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
