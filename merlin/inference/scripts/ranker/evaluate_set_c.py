"""Evaluate Full MERLIN and baselines on the reusable Set-C development set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from ...artifacts.integrity import sha256_path
from ...artifacts.paths import InferenceArtifactPaths
from ...recall.pool import load_candidate_pool_manifest
from ...evaluation.metrics import (
    macro_metrics,
    paired_bootstrap_cis,
    random_ranking_expectation,
    score_query,
)
from ...evaluation.protocol import (
    ARTIST_BOOTSTRAP_SAMPLES,
    EVALUATION_CUTOFFS,
    EVALUATION_SEED,
    PRIMARY_CUTOFF,
    QUERY_BOOTSTRAP_SAMPLES,
    ROBUSTNESS_CONFIGS,
    SCORERS,
    load_development_protocol,
)
from ...artifacts.io import parquet_rows, read_row_artifact, write_json_atomic
from ...ranking.features import FEATURE_ORDER, FILL_FEATURES, load_raw_feature_manifest
from ...ranking.model import LogisticRanker
from ...training.validation_groups import (
    VALIDATION_QUERY_GROUPS,
    load_validation_group_manifest,
)


CANDIDATE_COUNT_THRESHOLDS = (900, 800)


def parse_args() -> argparse.Namespace:
    paths = InferenceArtifactPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=paths.development_protocol)
    parser.add_argument("--candidate-pool", type=Path, default=paths.development_candidate_pool)
    parser.add_argument(
        "--candidate-pool-manifest",
        type=Path,
        default=paths.development_candidate_pool_manifest,
    )
    parser.add_argument("--groups-manifest", type=Path, default=paths.development_groups_manifest)
    parser.add_argument("--positives", type=Path, default=paths.development_positives)
    parser.add_argument("--validation-pairs", type=Path, default=paths.development_validation_pairs)
    parser.add_argument("--features", type=Path, default=paths.development_raw_features)
    parser.add_argument(
        "--features-manifest", type=Path, default=paths.development_raw_features_manifest
    )
    parser.add_argument("--output", type=Path, default=paths.development_evaluation_report)
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    return parser.parse_args()


def _candidate_count_distribution(counts: Iterable[int]) -> dict[str, object]:
    """Summarize deduplicated pool sizes without treating 1000 as a minimum."""
    values = sorted(int(count) for count in counts)
    if not values or values[0] <= 0:
        raise ValueError("eligible candidate counts must be positive")

    def nearest_rank(probability: float) -> int:
        index = max(0, math.ceil(probability * len(values)) - 1)
        return values[index]

    return {
        "query_count": len(values),
        "canonical_maximum": 1000,
        "minimum": values[0],
        "p10": nearest_rank(0.10),
        "median": nearest_rank(0.50),
        "p90": nearest_rank(0.90),
        "p99": nearest_rank(0.99),
        "maximum": values[-1],
        "mean": sum(values) / len(values),
        "below_thresholds": {
            str(threshold): {
                "count": sum(value < threshold for value in values),
                "fraction": sum(value < threshold for value in values) / len(values),
            }
            for threshold in CANDIDATE_COUNT_THRESHOLDS
        },
    }


def _protocol_parents(paths: InferenceArtifactPaths) -> dict[str, str]:
    return {
        "split_manifest": sha256_path(paths.split_manifest),
        "split_assignments": sha256_path(paths.split_assignments),
        "candidate_policy_manifest": sha256_path(paths.candidate_policy),
        "validation_group_thresholds": sha256_path(
            paths.validation_group_thresholds
        ),
        "ranker_training_manifest": sha256_path(paths.ranker_training_manifest),
        "no_hard_neg_training_manifest": sha256_path(
            paths.no_hard_neg_training_manifest
        ),
        "audio_index_manifest": sha256_path(paths.audio_manifest),
        "graph_index_manifest": sha256_path(paths.graph_manifest),
        "tag_idf": sha256_path(paths.tag_idf),
        "songs_metadata": sha256_path(paths.songs_metadata),
        "graph_edges": sha256_path(paths.graph_edges),
    }


def _load_ranker(root: Path, *, scope: str, variant: str) -> tuple[LogisticRanker, dict[str, float]]:
    schema = root / "ranker_feature_schema.json"
    scaler = root / "ranker_scaler.json"
    coefficients = root / "ranker_coefficients.json"
    manifest_path = root / "training_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("artifact_type") != "ranker_training"
        or manifest.get("scope") != scope
        or manifest.get("stage") != "final_retrain"
        or manifest.get("converged") is not True
    ):
        raise ValueError(f"{variant} ranker manifest is invalid")
    if manifest.get("selection", {}).get("training_variant", "full") != variant:
        raise ValueError(f"{variant} ranker training variant mismatch")
    hashes = manifest.get("artifact_hashes", {})
    for artifact in (schema, scaler, coefficients):
        if hashes.get(artifact.name) != sha256_path(artifact):
            raise ValueError(f"{variant} ranker artifact hash mismatch: {artifact.name}")
    with scaler.open("r", encoding="utf-8") as stream:
        scaler_payload = json.load(stream)
    fills = {name: float(value) for name, value in scaler_payload["fill_values"].items()}
    if set(fills) != set(FILL_FEATURES) or any(
        not math.isfinite(value) for value in fills.values()
    ):
        raise ValueError(f"{variant} ranker fill-value contract is invalid")
    ranker = LogisticRanker.from_artifacts(schema, scaler, coefficients)
    if ranker.feature_order != FEATURE_ORDER:
        raise ValueError(f"{variant} ranker feature order is not canonical")
    return ranker, fills


def _query_groups(path: Path) -> Iterable[tuple[str, list[dict[str, object]]]]:
    current_query = None
    current_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in read_row_artifact(path):
        query_id = str(row["query_track_id"])
        if current_query is None:
            current_query = query_id
        if query_id != current_query:
            if query_id in seen:
                raise ValueError("development feature rows are not clustered by query")
            seen.add(current_query)
            yield current_query, current_rows
            current_query = query_id
            current_rows = []
        current_rows.append(row)
    if current_query is not None:
        if current_query in seen:
            raise ValueError("development feature rows repeat a completed query")
        yield current_query, current_rows


def _candidate_metrics(rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    states = {
        group: {
            "eligible": None,
            "candidate_count": 0,
            "union_hits": 0,
            "source_hits": Counter(),
            "minus_hits": Counter(),
            "exclusive_hits": Counter(),
        }
        for group in VALIDATION_QUERY_GROUPS
    }
    sources = ("audio", "graph", "bfs", "tag")
    for row in rows:
        recalled_by = {str(source) for source in row.get("recall_sources", ())}
        primary_recalled_by = {
            str(source) for source in row.get("primary_recall_sources", ())
        }
        for membership in row["validation_groups"]:
            group = str(membership["query_group"])
            state = states.get(group)
            if state is None:
                continue
            state["candidate_count"] += 1
            denominator = int(membership["eligible_positive_count"])
            if state["eligible"] is None:
                state["eligible"] = denominator
            elif state["eligible"] != denominator:
                raise ValueError("candidate denominator changed within query")
            if int(membership["label"]) != 1:
                continue
            state["union_hits"] += 1
            state["source_hits"].update(primary_recalled_by)
            if len(recalled_by) == 1:
                state["exclusive_hits"].update(recalled_by)
            for source in sources:
                if recalled_by - {source}:
                    state["minus_hits"][source] += 1

    result = []
    for group in VALIDATION_QUERY_GROUPS:
        state = states[group]
        eligible = state["eligible"]
        if eligible is None:
            continue
        candidate_count = state["candidate_count"]
        union_hits = state["union_hits"]
        result.append({
            "query_group": group,
            "eligible_positive_count": eligible,
            "union_recall@1000": union_hits / eligible,
            "random_expectation": random_ranking_expectation(
                candidate_count,
                union_hits,
                eligible,
            ),
            "single_source_recall@250": {
                source: state["source_hits"][source] / eligible
                for source in sources
            },
            "all_minus_one_recall@1000": {
                source: state["minus_hits"][source] / eligible
                for source in sources
            },
            "exclusive_positive_hits": dict(state["exclusive_hits"]),
        })
    return result


def _aggregate_candidate(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    grouped = defaultdict(list)
    exclusive = defaultdict(Counter)
    for row in rows:
        group = str(row["query_group"])
        grouped[group].append(row)
        exclusive[group].update(row["exclusive_positive_hits"])
    report = {}
    for group in VALIDATION_QUERY_GROUPS:
        values = grouped[group]
        report[group] = {
            "query_count": len(values),
            "union_recall@1000": sum(float(row["union_recall@1000"]) for row in values)
            / len(values),
            "single_source_recall@250": {
                source: sum(float(row["single_source_recall@250"][source]) for row in values)
                / len(values)
                for source in ("audio", "graph", "bfs", "tag")
            },
            "all_minus_one_recall@1000": {
                source: sum(float(row["all_minus_one_recall@1000"][source]) for row in values)
                / len(values)
                for source in ("audio", "graph", "bfs", "tag")
            },
            "exclusive_positive_hits": dict(exclusive[group]),
        }
    return report


def _aggregate_random_expectation(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    expanded = []
    for index, row in enumerate(rows):
        expanded.append({
            "query_track_id": str(index),
            "query_group": row["query_group"],
            "scorer": "random_expectation",
            **row["random_expectation"],
        })
    return macro_metrics(expanded)


def _decade_slices(rows: Iterable[Mapping[str, object]], years: Mapping[str, int]):
    slices = defaultdict(list)
    missing = []
    for row in rows:
        if row["scorer"] != "full":
            continue
        year = years.get(str(row["query_track_id"]))
        target = missing if year is None else slices[f"{year // 10 * 10}s"]
        target.append(float(row[f"ndcg@{PRIMARY_CUTOFF}"]))
    return {
        **{
            decade: {"query_group_rows": len(values), "mean_ndcg@20": sum(values) / len(values)}
            for decade, values in sorted(slices.items())
        },
        "missing_year": {
            "query_group_rows": len(missing),
            "mean_ndcg@20": sum(missing) / len(missing) if missing else None,
        },
    }


def _catalog_report(
    metadata_path: Path,
    query_ids: set[str],
    top_counts: Mapping[int, Counter[str]],
) -> tuple[dict[str, str], dict[str, int], dict[str, object]]:
    import numpy as np

    popularity = []
    catalog_artists = set()
    catalog_tracks = 0
    query_artists = {}
    query_years = {}
    top_track_ids = set().union(*(counts for counts in top_counts.values()))
    top_metadata = {}
    for track, artist, song_popularity, year, has_year in parquet_rows(
        metadata_path,
        ("track_id", "artist_id", "song_hotttnesss", "year", "has_year"),
        engine="pyarrow",
    ):
        track_id = str(track)
        catalog_tracks += 1
        if artist:
            catalog_artists.add(str(artist))
        finite_pop = (
            float(song_popularity)
            if song_popularity is not None and math.isfinite(float(song_popularity))
            else None
        )
        if finite_pop is not None:
            popularity.append(finite_pop)
        if track_id in query_ids:
            query_artists[track_id] = (
                str(artist) if artist else f"missing:{track_id}"
            )
            if has_year and year is not None and int(year) > 0:
                query_years[track_id] = int(year)
        if track_id in top_track_ids:
            top_metadata[track_id] = (
                str(artist) if artist else None,
                finite_pop,
            )
    if not popularity:
        raise ValueError("catalog contains no finite popularity values")
    tail_threshold = float(np.quantile(np.asarray(popularity), 0.2, method="linear"))
    catalog_average = sum(popularity) / len(popularity)
    accumulators = {
        cutoff: {
            "artists": set(),
            "pop_sum": 0.0,
            "pop_count": 0,
            "tail_tracks": set(),
        }
        for cutoff in top_counts
    }
    catalog_tail_count = sum(value <= tail_threshold for value in popularity)
    for cutoff, counts in top_counts.items():
        for track_id, occurrences in counts.items():
            artist, finite_pop = top_metadata.get(track_id, (None, None))
            accumulator = accumulators[cutoff]
            if artist:
                accumulator["artists"].add(artist)
            if finite_pop is not None:
                accumulator["pop_sum"] += occurrences * finite_pop
                accumulator["pop_count"] += occurrences
                if finite_pop <= tail_threshold:
                    accumulator["tail_tracks"].add(track_id)
    coverage = {
        f"top_{cutoff}": {
            "catalog_track_coverage": len(counts) / catalog_tracks,
            "artist_coverage": len(accumulators[cutoff]["artists"]) / len(catalog_artists),
            "average_popularity": (
                accumulators[cutoff]["pop_sum"] / accumulators[cutoff]["pop_count"]
                if accumulators[cutoff]["pop_count"] else None
            ),
            "pop_lift_ratio": (
                accumulators[cutoff]["pop_sum"]
                / accumulators[cutoff]["pop_count"]
                / catalog_average
                if accumulators[cutoff]["pop_count"] and catalog_average else None
            ),
            "tail_coverage": len(accumulators[cutoff]["tail_tracks"]) / catalog_tail_count,
            "within_list_duplicate_rate": 0.0,
        }
        for cutoff, counts in sorted(top_counts.items())
    } | {
        "catalog": {
            "track_count": catalog_tracks,
            "artist_count": len(catalog_artists),
            "mean_song_popularity": catalog_average,
            "tail_quantile": 0.2,
            "tail_popularity_threshold": tail_threshold,
            "tail_track_count": catalog_tail_count,
        }
    }
    return query_artists, query_years, coverage


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"development report already exists: {args.output}")
    paths = InferenceArtifactPaths()
    protocol = load_development_protocol(
        args.protocol,
        expected_scope=args.scope,
        expected_split="set_c",
        expected_parent_hashes=_protocol_parents(paths),
    )
    candidate_manifest = load_candidate_pool_manifest(
        args.candidate_pool_manifest,
        args.candidate_pool,
        expected_scope=args.scope,
        expected_parent_hashes={"evaluation_protocol": sha256_path(args.protocol)},
    )
    group_manifest = load_validation_group_manifest(
        args.groups_manifest,
        thresholds_path=paths.validation_group_thresholds,
        positives_path=args.positives,
        validation_pairs_path=args.validation_pairs,
        expected_scope=args.scope,
        expected_apply_split="set_c",
        expected_parent_hashes={"evaluation_protocol": sha256_path(args.protocol)},
    )
    feature_manifest = load_raw_feature_manifest(
        args.features_manifest,
        args.features,
        expected_scope=args.scope,
        expected_pair_kind="validation",
        expected_stage="development_evaluation",
    )
    if feature_manifest.get("parent_hashes", {}).get(
        "evaluation_protocol"
    ) != sha256_path(args.protocol):
        raise ValueError("development features are not bound to the protocol")
    full, fills = _load_ranker(paths.ranker_coefficients.parent, scope=args.scope, variant="full")
    no_hard, no_hard_fills = _load_ranker(
        paths.no_hard_neg_coefficients.parent,
        scope=args.scope,
        variant="no_hard_neg",
    )
    if no_hard.feature_order != full.feature_order or no_hard_fills != fills:
        raise ValueError("Full and no-hard-neg models do not share frozen preprocessing")

    ranking_rows = []
    candidate_rows = []
    top_counts = {cutoff: Counter() for cutoff in EVALUATION_CUTOFFS}
    query_ids = set()
    eligible_candidate_counts = []
    for index, (query_id, rows) in enumerate(_query_groups(args.features), 1):
        query_ids.add(query_id)
        eligible_candidate_counts.append(len(rows))
        query_metrics, rankings = score_query(
            query_id,
            rows,
            full_ranker=full,
            no_hard_ranker=no_hard,
            fill_values=fills,
        )
        ranking_rows.extend(query_metrics)
        candidate_rows.extend(_candidate_metrics(rows))
        for cutoff in EVALUATION_CUTOFFS:
            top_counts[cutoff].update(rankings["full"][:cutoff])
        if index % 256 == 0:
            print(f"development_evaluation_progress queries={index}", flush=True)
    if len(query_ids) > int(candidate_manifest["query_count"]):
        raise ValueError("development queries exceed the candidate pool")
    expected_scorers = set(SCORERS) | set(ROBUSTNESS_CONFIGS)
    if {str(row["scorer"]) for row in ranking_rows} != expected_scorers:
        raise ValueError("development evaluation did not run every scorer")

    artists, years, coverage = _catalog_report(
        paths.songs_metadata,
        query_ids,
        top_counts,
    )
    rows_by_scorer = defaultdict(list)
    for row in ranking_rows:
        rows_by_scorer[str(row["scorer"])].append(row)
    ranking_report = {
        scorer: macro_metrics(rows_by_scorer[scorer])
        for scorer in sorted(expected_scorers)
    }
    ranking_report["random_expectation"] = _aggregate_random_expectation(
        candidate_rows
    )
    baselines = tuple(scorer for scorer in SCORERS if scorer != "full")
    query_bootstraps = paired_bootstrap_cis(
        ranking_rows,
        baselines=baselines,
        metric=f"ndcg@{PRIMARY_CUTOFF}",
        samples=QUERY_BOOTSTRAP_SAMPLES,
    )
    artist_bootstraps = paired_bootstrap_cis(
        ranking_rows,
        baselines=baselines,
        metric=f"ndcg@{PRIMARY_CUTOFF}",
        samples=ARTIST_BOOTSTRAP_SAMPLES,
        clusters=artists,
    )
    inference = {
        baseline: {
            "query_bootstrap": query_bootstraps[baseline],
            "artist_cluster_bootstrap": artist_bootstraps[baseline],
        }
        for baseline in baselines
    }
    split_count = int(json.loads(paths.split_manifest.read_text())["track_counts"]["set_c"])
    report = {
        "artifact_type": "development_evaluation",
        "artifact_version": protocol["artifact_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": args.scope,
        "claims": protocol["claims"],
        "primary_metric": f"three_strata_macro_ndcg@{PRIMARY_CUTOFF}",
        "cutoffs": list(EVALUATION_CUTOFFS),
        "evaluated_query_count": len(query_ids),
        "development_track_count": split_count,
        "queries_without_any_eligible_group": split_count - len(query_ids),
        "group_eligibility": group_manifest["group_stats"],
        "candidate_layer": _aggregate_candidate(candidate_rows),
        "ranking": ranking_report,
        "paired_inference_full_minus_baseline": inference,
        "robustness": {
            "coverage_popularity_tail": coverage,
            "precomputed_acoustic_cold": ranking_report[
                "precomputed_acoustic_cold"
            ],
            "decade_slices": _decade_slices(ranking_rows, years),
            "eligible_query_candidate_counts": _candidate_count_distribution(
                eligible_candidate_counts
            ),
            "year_semantics": "MSD static year; not first-release time",
        },
        "lineage": {
            "protocol_sha256": sha256_path(args.protocol),
            "candidate_pool_manifest_sha256": sha256_path(args.candidate_pool_manifest),
            "validation_groups_manifest_sha256": sha256_path(args.groups_manifest),
            "raw_features_manifest_sha256": sha256_path(args.features_manifest),
            "full_ranker_manifest_sha256": sha256_path(paths.ranker_training_manifest),
            "no_hard_neg_ranker_manifest_sha256": sha256_path(
                paths.no_hard_neg_training_manifest
            ),
        },
    }
    write_json_atomic(report, args.output)
    print(
        f"development_evaluation_ready queries={len(query_ids)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
