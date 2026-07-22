"""Persist and validate the canonical Recall-to-Ranker handoff."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .artifact_lineage import artifact_size_bytes, sha256_path
from .candidate_policy import CANDIDATE_POLICY_VERSION
from .jsonl_artifact import read_row_artifact, write_json_atomic, write_row_artifact
from .recall import RecallPipeline
from .types import Candidate


CANDIDATE_POOL_VERSION = "merlin_candidate_pool_v2"
CANDIDATE_BATCH_SIZE = 256


def _candidate_payload(candidate: Candidate) -> dict[str, object]:
    return {
        "track_id": candidate.track_id,
        "recall_sources": sorted(candidate.sources),
    }


def export_candidate_pool(
    pipeline: RecallPipeline,
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
    source_totals = {name: 0 for name in pipeline.retriever_limits}

    def rows() -> Iterator[dict[str, object]]:
        for start in range(0, len(queries), CANDIDATE_BATCH_SIZE):
            batch = queries[start : start + CANDIDATE_BATCH_SIZE]
            recalled = pipeline.recall_many(batch)
            for query_id in batch:
                candidates, audit = recalled[query_id]
                totals["raw_candidates"] += audit.raw_candidates
                totals["unique_candidates"] += audit.unique_candidates
                for name, count in audit.source_counts.items():
                    source_totals[name] += count
                yield {
                    "query_track_id": query_id,
                    "candidates": [_candidate_payload(candidate) for candidate in candidates],
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
    row_count = write_row_artifact(rows(), output, parquet_schema=parquet_schema)
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
            "candidates": "ordered array<track_id, recall_sources>",
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


def iter_candidate_pool(path: str | Path) -> Iterator[dict[str, object]]:
    yield from read_row_artifact(path)
