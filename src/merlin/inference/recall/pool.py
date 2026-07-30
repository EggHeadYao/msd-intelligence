"""Persist and validate the canonical Recall-to-Ranker handoff."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import fmean
from typing import Iterable, Iterator, Mapping

from ..artifacts.integrity import artifact_size_bytes, sha256_path
from ..artifacts.io import read_row_artifact, write_json_atomic, write_row_artifact
from ..training.weak_labels import POSITIVE_SOURCES
from ..types import Candidate
from .pipeline import RecallPipeline
from .policy import CANDIDATE_POLICY_VERSION
from .streaming import SOURCE_NAMES, StreamingRecallEngine


CANDIDATE_POOL_VERSION = "merlin_candidate_pool_v3"
CANDIDATE_BATCH_SIZE = 256
CANDIDATE_READ_BATCH_SIZE = 64
CANDIDATE_AUDIT_VERSION = "merlin_candidate_audit_v1"
RECALL_SOURCES = ("audio", "graph", "bfs", "tag")


def _candidate_payload(
    candidate: Candidate,
    primary_limits: Mapping[str, int],
) -> dict[str, object]:
    return {
        "track_id": candidate.track_id,
        "recall_sources": sorted(candidate.sources),
        "primary_recall_sources": sorted(
            source
            for source, rank in candidate.source_ranks.items()
            if int(rank) <= int(primary_limits[source])
        ),
    }


def export_candidate_pool(
    pipeline: RecallPipeline | StreamingRecallEngine,
    query_track_ids: Iterable[str],
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    parent_paths: Mapping[str, str | Path],
    scope: str = "formal",
) -> dict[str, object]:
    """Write one ordered candidate-list record per query and bind all parents."""
    if scope not in {"formal", "smoke"}:
        raise ValueError("candidate pool scope must be formal or smoke")
    queries = tuple(query_track_ids)
    if not queries or any(not query_id for query_id in queries):
        raise ValueError("candidate pool queries must be non-empty")
    if len(set(queries)) != len(queries):
        raise ValueError("candidate pool queries must be unique")

    totals = {"raw_candidates": 0, "unique_candidates": 0}
    limits = (
        pipeline.limits
        if isinstance(pipeline, StreamingRecallEngine)
        else pipeline.retriever_limits
    )
    source_totals = {name: 0 for name in limits}

    def rows() -> Iterator[dict[str, object]]:
        for start in range(0, len(queries), CANDIDATE_BATCH_SIZE):
            batch = queries[start : start + CANDIDATE_BATCH_SIZE]
            streaming = isinstance(pipeline, StreamingRecallEngine)
            recalled = (
                pipeline.search_candidates_many(batch)
                if streaming
                else pipeline.recall_many(batch)
            )
            for position, query_id in enumerate(batch):
                candidates, audit = (
                    pipeline.candidate_query(recalled, position)
                    if streaming
                    else recalled[query_id]
                )
                totals["raw_candidates"] += audit.raw_candidates
                totals["unique_candidates"] += audit.unique_candidates
                for name, count in audit.source_counts.items():
                    source_totals[name] += count
                yield {
                    "query_track_id": query_id,
                    "candidates": (
                        [
                            {
                                "track_id": candidates.track_id(index),
                                "recall_sources": sorted(
                                    name
                                    for source, name in enumerate(SOURCE_NAMES)
                                    if int(candidates.source_masks[index]) & (1 << source)
                                ),
                                "primary_recall_sources": sorted(
                                    candidates.primary_sources(index)
                                ),
                            }
                            for index in range(len(candidates))
                        ]
                        if streaming
                        else [
                            _candidate_payload(candidate, limits)
                            for candidate in candidates
                        ]
                    ),
                    "audit": {
                        "raw_candidates": audit.raw_candidates,
                        "unique_candidates": audit.unique_candidates,
                        "duplicate_candidates": audit.duplicate_candidates,
                        "source_available": dict(audit.source_available),
                        "source_counts": dict(audit.source_counts),
                        "source_shortages": dict(audit.source_shortages),
                    },
                }
            processed = min(start + len(batch), len(queries))
            if processed == len(queries) or processed % (10 * CANDIDATE_BATCH_SIZE) == 0:
                print(
                    f"candidate_pool_progress queries={processed}/{len(queries)}",
                    flush=True,
                )

    output = Path(output_path)
    parquet_schema = None
    if output.suffix == ".parquet":
        import pyarrow as pa

        parquet_schema = pa.schema((
            pa.field("query_track_id", pa.string(), nullable=False),
            pa.field("candidates", pa.list_(pa.struct((
                pa.field("track_id", pa.string(), nullable=False),
                pa.field("recall_sources", pa.list_(pa.string()), nullable=False),
                pa.field(
                    "primary_recall_sources",
                    pa.list_(pa.string()),
                    nullable=False,
                ),
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
    row_count = write_row_artifact(
        rows(),
        output,
        parquet_schema=parquet_schema,
        batch_size=CANDIDATE_BATCH_SIZE,
    )
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
        "output_size_bytes": artifact_size_bytes(output),
        "parent_hashes": parents,
        "schema": {
            "query_track_id": "string",
            "candidates": (
                "ordered array<track_id, recall_sources, primary_recall_sources>"
            ),
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


def iter_candidate_pool(
    path: str | Path,
    *,
    batch_size: int = CANDIDATE_READ_BATCH_SIZE,
) -> Iterator[dict[str, object]]:
    yield from read_row_artifact(path, batch_size=batch_size)


def audit_candidate_pool(
    candidate_pool_path: str | Path,
    positives: Mapping[str, Mapping[str, frozenset[str]]],
) -> dict[str, object]:
    """Measure positive coverage and source attribution for one candidate pool."""
    query_reports = []
    for row in iter_candidate_pool(candidate_pool_path):
        query_id = str(row["query_track_id"])
        if query_id not in positives:
            raise ValueError(f"candidate query has no weak-positive record: {query_id}")
        positive_map = positives[query_id]
        if not positive_map:
            query_reports.append({"query_track_id": query_id, "eligible": False})
            continue
        candidate_sources = {
            str(candidate["track_id"]): frozenset(candidate["recall_sources"])
            for candidate in row["candidates"]
        }
        primary_candidate_sources = {
            str(candidate["track_id"]): frozenset(
                candidate["primary_recall_sources"]
            )
            for candidate in row["candidates"]
        }
        positive_ids = set(positive_map)
        union_hit_count = 0
        single = {source: 0 for source in RECALL_SOURCES}
        minus = {source: 0 for source in RECALL_SOURCES}
        exclusive = {source: 0 for source in RECALL_SOURCES}
        for track_id in positive_ids:
            sources = candidate_sources.get(track_id)
            if not sources:
                continue
            union_hit_count += 1
            primary_sources = primary_candidate_sources[track_id]
            for source in RECALL_SOURCES:
                single[source] += int(source in primary_sources)
                minus[source] += int(bool(sources - {source}))
                exclusive[source] += int(sources == frozenset({source}))
        strata = {
            source: {
                "positive_count": sum(
                    source in sources for sources in positive_map.values()
                ),
                "union_hits": sum(
                    source in sources and track_id in candidate_sources
                    for track_id, sources in positive_map.items()
                ),
            }
            for source in POSITIVE_SOURCES
        }
        denominator = len(positive_ids)
        query_reports.append({
            "query_track_id": query_id,
            "eligible": True,
            "positive_count": denominator,
            "union_recall": union_hit_count / denominator,
            "single_source_recall": {
                source: hits / denominator for source, hits in single.items()
            },
            "all_minus_source_recall": {
                source: hits / denominator for source, hits in minus.items()
            },
            "all_minus_source_delta": {
                source: (union_hit_count - hits) / denominator
                for source, hits in minus.items()
            },
            "exclusive_positive_hits": exclusive,
            "positive_strata": strata,
        })
    eligible = [report for report in query_reports if report.get("eligible")]
    if not eligible:
        raise ValueError("candidate audit has no query with eligible positives")
    return {
        "artifact_type": "candidate_audit",
        "artifact_version": CANDIDATE_AUDIT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_count": len(query_reports),
        "eligible_query_count": len(eligible),
        "no_positive_query_count": len(query_reports) - len(eligible),
        "macro_union_recall": fmean(float(row["union_recall"]) for row in eligible),
        "macro_single_source_recall": {
            source: fmean(
                float(row["single_source_recall"][source]) for row in eligible
            )
            for source in RECALL_SOURCES
        },
        "macro_all_minus_source_delta": {
            source: fmean(
                float(row["all_minus_source_delta"][source]) for row in eligible
            )
            for source in RECALL_SOURCES
        },
        "queries": query_reports,
    }


def write_candidate_audit(
    report: Mapping[str, object],
    output_path: str | Path,
    *,
    candidate_pool_path: str | Path,
    weak_positives_path: str | Path,
) -> dict[str, object]:
    payload = dict(report)
    payload["parent_hashes"] = {
        "candidate_pool": sha256_path(candidate_pool_path),
        "weak_positives": sha256_path(weak_positives_path),
    }
    write_json_atomic(payload, output_path)
    return payload
