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
