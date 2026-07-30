"""Deterministic song-group-aware C3 supervised split artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from ..artifacts.integrity import sha256_path
from ..artifacts.io import read_row_artifact, write_json_atomic, write_row_artifact


SPLIT_VERSION = "merlin_group_split_v4"
SPLIT_SEED = 42
SPLIT_RANGES = (
    ("set_a", "none", 0.00, 0.10),
    ("set_b", "tune", 0.10, 0.11),
    ("set_c", "none", 0.11, 0.13),
    ("set_b", "confirm", 0.15, 0.17),
)


def split_key(track_id: str, song_id: str | None) -> str:
    if not track_id:
        raise ValueError("split track_id must not be empty")
    return song_id if song_id else track_id


def assign_split_and_fold(
    key: str, seed: int = SPLIT_SEED
) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    for name, fold, lower, upper in SPLIT_RANGES:
        if lower <= value < upper:
            return name, fold
    return "remaining", "none"


def assign_split(key: str, seed: int = SPLIT_SEED) -> str:
    return assign_split_and_fold(key, seed)[0]


def build_split_artifacts(
    rows: Iterable[tuple[str, str | None]],
    assignments_path: str | Path,
    manifest_path: str | Path,
    *,
    songs_metadata_path: str | Path,
    scope: str = "formal",
) -> dict[str, object]:
    if scope not in {"formal", "smoke"}:
        raise ValueError("split scope must be formal or smoke")
    counts: Counter[str] = Counter()
    fold_counts: Counter[str] = Counter()
    group_assignments: dict[str, tuple[str, str]] = {}
    seen_tracks: set[str] = set()

    def assignments() -> Iterator[dict[str, object]]:
        for track_id, song_id in rows:
            if not track_id:
                raise ValueError("split input contains an empty track_id")
            if track_id in seen_tracks:
                raise ValueError(f"split input contains duplicate track {track_id!r}")
            seen_tracks.add(track_id)
            key = split_key(track_id, song_id)
            assignment, selection_fold = assign_split_and_fold(key)
            previous = group_assignments.setdefault(
                key, (assignment, selection_fold)
            )
            if previous != (assignment, selection_fold):
                raise ValueError("one split group crosses a set or selection fold")
            counts[assignment] += 1
            if selection_fold != "none":
                fold_counts[selection_fold] += 1
            yield {
                "track_id": track_id,
                "split_key": key,
                "split": assignment,
                "selection_fold": selection_fold,
            }

    output = Path(assignments_path)
    parquet_schema = None
    if output.suffix == ".parquet":
        import pyarrow as pa

        parquet_schema = pa.schema((
            pa.field("track_id", pa.string(), nullable=False),
            pa.field("split_key", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("selection_fold", pa.string(), nullable=False),
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
        "proportions": {
            "set_a": 0.10,
            "set_b": 0.03,
            "set_c": 0.02,
            "remaining": 0.85,
        },
        "set_b_selection_folds": {"tune": 0.01, "confirm": 0.02},
        "hash_ranges": [
            {"split": name, "selection_fold": fold, "lower": lower, "upper": upper}
            for name, fold, lower, upper in SPLIT_RANGES
        ],
        "track_count": row_count,
        "group_count": len(group_assignments),
        "track_counts": dict(sorted(counts.items())),
        "set_b_selection_fold_counts": dict(sorted(fold_counts.items())),
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
    expected_ranges = [
        {"split": name, "selection_fold": fold, "lower": lower, "upper": upper}
        for name, fold, lower, upper in SPLIT_RANGES
    ]
    if manifest.get("hash_ranges") != expected_ranges:
        raise ValueError("split hash ranges changed")
    if manifest.get("set_b_selection_folds") != {
        "tune": 0.01,
        "confirm": 0.02,
    }:
        raise ValueError("Set-B selection-fold proportions changed")
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
    group_assignments: dict[str, tuple[str, str]] = {}
    for row in read_row_artifact(path):
        track_id = str(row["track_id"])
        key = str(row["split_key"])
        split = str(row["split"])
        selection_fold = str(row.get("selection_fold", "none"))
        if split not in {"set_a", "set_b", "set_c", "remaining"}:
            raise ValueError("split assignment contains an unknown set")
        if selection_fold not in {"none", "tune", "confirm"}:
            raise ValueError("split assignment contains an unknown selection fold")
        if (split == "set_b") != (selection_fold in {"tune", "confirm"}):
            raise ValueError("selection fold is inconsistent with the assigned split")
        if track_id in assignments:
            raise ValueError("split assignments contain a duplicate track")
        group_assignment = (split, selection_fold)
        if key in group_assignments and group_assignments[key] != group_assignment:
            raise ValueError("one split group crosses a set or selection fold")
        assignments[track_id] = split
        group_assignments[key] = group_assignment
    if not assignments:
        raise ValueError("split assignments must not be empty")
    return assignments
