    def recall(self, query_track_id: str) -> tuple[list[Candidate], RecallAudit]:
        return recall_candidates(
            self.retrievers,
            self.retriever_limits,
            self.candidate_limit,
            query_track_id,
        )


def candidate_digest(candidates: Sequence[Candidate]) -> str:
    payload = [
        {
            "track_id": candidate.track_id,
            "sources": sorted(candidate.sources),
            "recall_scores": dict(sorted(candidate.recall_scores.items())),
            "source_ranks": dict(sorted(candidate.source_ranks.items())),
        }
        for candidate in candidates
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def recall_query_report(
    query_id: str,
    candidates: Sequence[Candidate],
    audit: RecallAudit,
) -> dict[str, object]:
    return {
        "query_track_id": query_id,
        "candidate_digest_sha256": candidate_digest(candidates),
        "raw_candidates": audit.raw_candidates,
        "unique_candidates": audit.unique_candidates,
        "duplicate_candidates": audit.duplicate_candidates,
        "deduplication_rate": audit.deduplication_rate,
        "source_available": dict(audit.source_available),
        "source_counts": dict(audit.source_counts),
        "source_shortages": dict(audit.source_shortages),
        "exclusive_candidates": dict(audit.exclusive_candidates),
    }


def validate_recall_pipeline(
    pipeline: RecallPipeline,
    query_track_ids: Iterable[str],
) -> dict[str, object]:
    """Repeat fixed queries and report deterministic structural recall coverage."""
    reports: list[dict[str, object]] = []
    for query_id in query_track_ids:
        first, audit = pipeline.recall(query_id)
        second, repeated_audit = pipeline.recall(query_id)
        first_digest = candidate_digest(first)
        if first_digest != candidate_digest(second) or audit != repeated_audit:
            raise ValueError(f"recall is not deterministic for {query_id}")
        reports.append(recall_query_report(query_id, first, audit))
    if not reports:
        raise ValueError("recall validation requires at least one query")
    return {
        "validation_status": "PASS",
        "validation_type": "structural_recall_audit",
        "candidate_recall_metrics_available": False,
        "query_count": len(reports),
        "queries": reports,
    }


def write_recall_report(report: Mapping[str, object], path: str | Path) -> None:
    """Atomically publish a structural recall validation report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(report), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)
