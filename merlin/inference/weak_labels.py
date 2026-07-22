"""Frozen weak-positive thresholds, selection, and artifact contracts."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from itertools import islice
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from .artifact_lineage import sha256_path
from .jsonl_artifact import read_row_artifact, write_json_atomic, write_row_artifact


WEAK_LABEL_VERSION = "merlin_weak_labels_v1"
WEAK_LABEL_SEED = 42
MAX_THRESHOLD_PAIRS = 1_000_000
MAX_POSITIVES_PER_QUERY = 50
POSITIVE_SOURCES = ("same_artist", "tag_derived", "audio_derived")
PairSimilarity = Callable[[str, str], float | None]
PairBatchSimilarity = Callable[[Sequence[tuple[str, str]]], Sequence[float | None]]


def deterministic_cross_artist_pairs(
    track_ids: Sequence[str],
    track_to_artist: Mapping[str, str],
    *,
    max_pairs: int = MAX_THRESHOLD_PAIRS,
    seed: int = WEAK_LABEL_SEED,
) -> Iterator[tuple[str, str]]:
    """Yield a stable bounded sample without materializing the pair product."""
    tracks = tuple(sorted(set(track_ids)))
    if len(tracks) < 2 or max_pairs <= 0:
        return
    target = min(max_pairs, len(tracks) * max(1, min(20, len(tracks) - 1)))
    seen: set[tuple[str, str]] = set()
    attempts = 0
    maximum_attempts = max(target * 20, len(tracks) * 4)
    cursor = 0
    while len(seen) < target and attempts < maximum_attempts:
        left = tracks[cursor % len(tracks)]
        digest = hashlib.sha256(
            f"{seed}\0{left}\0{attempts}".encode("utf-8")
        ).digest()
        right = tracks[int.from_bytes(digest[:8], "big") % len(tracks)]
        attempts += 1
        cursor += 1
        if left == right:
            continue
        left_artist = track_to_artist.get(left)
        right_artist = track_to_artist.get(right)
        if left_artist is None or right_artist is None or left_artist == right_artist:
            continue
        pair = tuple(sorted((left, right)))
        if pair in seen:
            continue
        seen.add(pair)
        yield pair


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 < quantile <= 1.0:
        raise ValueError("quantile input must be non-empty and in (0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def fit_weak_label_thresholds(
    track_ids: Sequence[str],
    track_to_artist: Mapping[str, str],
    audio_similarity: PairSimilarity,
    tag_similarity: PairSimilarity,
    *,
    max_pairs: int = MAX_THRESHOLD_PAIRS,
    audio_batch_similarity: PairBatchSimilarity | None = None,
    batch_size: int = 10_000,
) -> dict[str, object]:
    """Fit both p90 values once from deterministic Set-A cross-artist pairs."""
    audio_values: list[float] = []
    tag_values: list[float] = []
    sampled = 0
    if batch_size <= 0:
        raise ValueError("weak-label similarity batch size must be positive")
    pairs = deterministic_cross_artist_pairs(
        track_ids,
        track_to_artist,
        max_pairs=max_pairs,
    )
    while batch := list(islice(pairs, batch_size)):
        audio_scores = (
            audio_batch_similarity(batch)
            if audio_batch_similarity is not None
            else [audio_similarity(left, right) for left, right in batch]
        )
        if len(audio_scores) != len(batch):
            raise ValueError("audio batch similarity returned the wrong number of scores")
        for (left, right), audio in zip(batch, audio_scores, strict=True):
            sampled += 1
            tag = tag_similarity(left, right)
            if audio is not None and math.isfinite(float(audio)):
                audio_values.append(float(audio))
            if tag is not None and math.isfinite(float(tag)) and float(tag) > 0.0:
                tag_values.append(float(tag))
    if not audio_values:
        raise ValueError("Set-A threshold sample has no valid audio similarities")
    if not tag_values:
        raise ValueError("Set-A threshold sample has no nonzero tag similarities")
    return {
        "artifact_type": "weak_label_thresholds",
        "artifact_version": WEAK_LABEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_split": "set_a",
        "seed": WEAK_LABEL_SEED,
        "quantile": 0.90,
        "quantile_method": "nearest_rank",
        "max_sample_pairs": max_pairs,
        "sampled_cross_artist_pairs": sampled,
        "valid_audio_pairs": len(audio_values),
        "nonzero_tag_pairs": len(tag_values),
        "audio_cosine_p90": _nearest_rank(audio_values, 0.90),
        "tag_tfidf_cosine_p90": _nearest_rank(tag_values, 0.90),
    }


def select_weak_positives(
    query_track_id: str,
    allowed_tracks: set[str],
    track_to_artist: Mapping[str, str],
    same_artist_tracks: Sequence[str],
    audio_neighbors: Sequence[tuple[str, float]],
    tag_neighbors: Sequence[tuple[str, float]],
    same_song: Callable[[str, str], bool],
    thresholds: Mapping[str, object],
    *,
    limit: int = MAX_POSITIVES_PER_QUERY,
) -> list[dict[str, object]]:
    """Apply the three predicates and cap with deterministic source round-robin."""
    if limit <= 0:
        raise ValueError("positive limit must be positive")
    root_artist = track_to_artist.get(query_track_id)
    audio_threshold = float(thresholds["audio_cosine_p90"])
    tag_threshold = float(thresholds["tag_tfidf_cosine_p90"])
    provenance: dict[str, set[str]] = {}

    def eligible(track_id: str) -> bool:
        return (
            track_id in allowed_tracks
            and track_id != query_track_id
            and not same_song(query_track_id, track_id)
        )

    same_artist = sorted(
        track_id for track_id in same_artist_tracks if eligible(track_id)
    )
    audio = sorted(
        (
            (track_id, float(score))
            for track_id, score in audio_neighbors
            if eligible(track_id)
            and track_to_artist.get(track_id) != root_artist
            and math.isfinite(float(score))
            and float(score) >= audio_threshold
        ),
        key=lambda item: (-item[1], item[0]),
    )
    tags = sorted(
        (
            (track_id, float(score))
            for track_id, score in tag_neighbors
            if eligible(track_id)
            and track_to_artist.get(track_id) != root_artist
            and math.isfinite(float(score))
            and float(score) >= tag_threshold
        ),
        key=lambda item: (-item[1], item[0]),
    )
    source_lists = {
        "same_artist": same_artist,
        "tag_derived": [track_id for track_id, _score in tags],
        "audio_derived": [track_id for track_id, _score in audio],
    }
    for source, tracks in source_lists.items():
        for track_id in tracks:
            provenance.setdefault(track_id, set()).add(source)

    positions = {source: 0 for source in POSITIVE_SOURCES}
    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < limit:
        progressed = False
        for source in POSITIVE_SOURCES:
            tracks = source_lists[source]
            while positions[source] < len(tracks):
                track_id = tracks[positions[source]]
                positions[source] += 1
                if track_id in selected_set:
                    continue
                selected.append(track_id)
                selected_set.add(track_id)
                progressed = True
                break
            if len(selected) == limit:
                break
        if not progressed:
            break
    return [
        {"track_id": track_id, "positive_sources": sorted(provenance[track_id])}
        for track_id in selected
    ]


def write_weak_positive_artifacts(
    records: Iterable[Mapping[str, object]],
    positives_path: str | Path,
    manifest_path: str | Path,
    thresholds_path: str | Path,
    thresholds: Mapping[str, object],
    *,
    parent_paths: Mapping[str, str | Path],
    scope: str,
) -> dict[str, object]:
    if scope not in {"formal", "smoke"}:
        raise ValueError("weak-positive scope must be formal or smoke")
    write_json_atomic(thresholds, thresholds_path)
    query_count = 0
    positive_count = 0
    source_counts: Counter[str] = Counter()

    def counted() -> Iterator[Mapping[str, object]]:
        nonlocal query_count, positive_count
        for record in records:
            positives = record.get("positives")
            if not isinstance(positives, list):
                raise ValueError("weak-positive record is missing positives")
            query_count += 1
            positive_count += len(positives)
            for positive in positives:
                for source in positive["positive_sources"]:
                    source_counts[str(source)] += 1
            yield record

    output = Path(positives_path)
    parquet_schema = None
    if output.suffix == ".parquet":
        import pyarrow as pa

        parquet_schema = pa.schema((
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
