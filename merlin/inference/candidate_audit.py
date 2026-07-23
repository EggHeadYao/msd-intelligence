"""Positive-aware Candidate Recall audit for the canonical four-source pool."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Mapping

from .artifact_lineage import sha256_path
from .candidate_pool import iter_candidate_pool
from .jsonl_artifact import write_json_atomic
from .training.weak_labels import POSITIVE_SOURCES


CANDIDATE_AUDIT_VERSION = "merlin_candidate_audit_v1"
RECALL_SOURCES = ("audio", "graph", "bfs", "tag")


def audit_candidate_pool(
    candidate_pool_path: str | Path,
    positives: Mapping[str, Mapping[str, frozenset[str]]],
) -> dict[str, object]:
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
            for source in RECALL_SOURCES:
                single[source] += int(source in sources)
                minus[source] += int(bool(sources - {source}))
                exclusive[source] += int(sources == frozenset({source}))
        strata = {
            source: {
                "positive_count": sum(source in sources for sources in positive_map.values()),
                "union_hits": sum(
                    source in sources and track_id in candidate_sources
                    for track_id, sources in positive_map.items()
                ),
            }
            for source in POSITIVE_SOURCES
        }
        denominator = len(positive_ids)
        query_reports.append(
            {
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
            }
        )
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
