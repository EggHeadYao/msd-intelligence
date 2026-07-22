"""Determinism and coverage validation for a constructed C3 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .pipeline import MerlinPipeline


def validate_pipeline(
    pipeline: MerlinPipeline,
    query_track_ids: Iterable[str],
    *,
    score_tolerance: float = 1e-7,
) -> dict[str, object]:
    """Repeat fixed queries and return a PASS report only on exact ID stability."""
    if score_tolerance < 0.0:
        raise ValueError("score_tolerance must be non-negative")
    query_reports = []
    for query_id in query_track_ids:
        first_candidates, audit = pipeline.recall(query_id)
        second_candidates, _ = pipeline.recall(query_id)
        first_ids = [candidate.track_id for candidate in first_candidates]
        second_ids = [candidate.track_id for candidate in second_candidates]
        if first_ids != second_ids:
            raise ValueError(f"candidate order is not deterministic for {query_id}")

        first = pipeline.recommend(query_id)
        second = pipeline.recommend(query_id)
        if [item.track_id for item in first] != [item.track_id for item in second]:
            raise ValueError(f"Top-K IDs are not deterministic for {query_id}")
        max_error = max(
            (abs(left.relevance_score - right.relevance_score)
             for left, right in zip(first, second, strict=True)),
            default=0.0,
        )
        if max_error > score_tolerance:
            raise ValueError(f"Top-K scores are not deterministic for {query_id}")
        query_reports.append({
            "query_track_id": query_id,
            "raw_candidates": audit.raw_candidates,
            "unique_candidates": audit.unique_candidates,
            "duplicate_candidates": audit.duplicate_candidates,
            "deduplication_rate": audit.deduplication_rate,
            "source_counts": dict(audit.source_counts),
            "source_shortages": dict(audit.source_shortages),
            "source_available": dict(audit.source_available),
            "exclusive_candidates": dict(audit.exclusive_candidates),
            "result_track_ids": [item.track_id for item in first],
            "max_repeat_score_error": max_error,
        })
    if not query_reports:
        raise ValueError("inference validation requires at least one query")
    return {
        "validation_status": "PASS",
        "query_count": len(query_reports),
        "score_tolerance": score_tolerance,
        "queries": query_reports,
    }


def write_validation_report(report: dict[str, object], path: str | Path) -> None:
    """Atomically publish a completed validation report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)
