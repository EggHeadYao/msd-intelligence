"""Evaluate frozen Full MERLIN and baselines once on canonical Set-C candidates."""

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
    paired_bootstrap_ci,
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
    load_set_c_protocol,
)
from ...artifacts.io import parquet_rows, read_row_artifact, write_json_atomic
from ...ranking.features import FEATURE_ORDER, FILL_FEATURES, load_raw_feature_manifest
from ...ranking.model import LogisticRanker
from ...training.validation_groups import (
    VALIDATION_QUERY_GROUPS,
    load_validation_group_manifest,
)


def parse_args() -> argparse.Namespace:
    paths = InferenceArtifactPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=paths.set_c_protocol)
    parser.add_argument("--candidate-pool", type=Path, default=paths.set_c_candidate_pool)
    parser.add_argument(
        "--candidate-pool-manifest",
        type=Path,
        default=paths.set_c_candidate_pool_manifest,
    )
    parser.add_argument("--groups-manifest", type=Path, default=paths.set_c_groups_manifest)
    parser.add_argument("--positives", type=Path, default=paths.set_c_positives)
    parser.add_argument("--validation-pairs", type=Path, default=paths.set_c_validation_pairs)
    parser.add_argument("--features", type=Path, default=paths.set_c_raw_features)
    parser.add_argument(
        "--features-manifest", type=Path, default=paths.set_c_raw_features_manifest
    )
    parser.add_argument("--output", type=Path, default=paths.set_c_evaluation_report)
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    return parser.parse_args()


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
                raise ValueError("Set-C feature rows are not clustered by query")
            seen.add(current_query)
            yield current_query, current_rows
            current_query = query_id
            current_rows = []
        current_rows.append(row)
    if current_query is not None:
        if current_query in seen:
            raise ValueError("Set-C feature rows repeat a completed query")
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
            state["source_hits"].update(recalled_by)
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


def _query_metadata(path: Path, query_ids: set[str]) -> tuple[dict[str, str], dict[str, int]]:
    artists = {}
    years = {}
    for track_id, artist_id, year, has_year in parquet_rows(
        path, ("track_id", "artist_id", "year", "has_year"), engine="pyarrow"
    ):
        key = str(track_id)
        if key not in query_ids:
            continue
        artists[key] = str(artist_id) if artist_id else f"missing:{key}"
        if has_year and year is not None and int(year) > 0:
            years[key] = int(year)
    return artists, years


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


def _coverage_report(
    metadata_path: Path,
    top_counts: Mapping[int, Counter[str]],
) -> dict[str, object]:
    import numpy as np

    popularity = []
    catalog_artists = set()
    catalog_tracks = 0
    for _track, artist, song_popularity in parquet_rows(
        metadata_path,
        ("track_id", "artist_id", "song_hotttnesss"),
        engine="pyarrow",
    ):
        catalog_tracks += 1
        if artist:
            catalog_artists.add(str(artist))
        if song_popularity is not None and math.isfinite(float(song_popularity)):
            popularity.append(float(song_popularity))
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
    catalog_tail_count = 0
    for track, artist, song_popularity in parquet_rows(
        metadata_path,
        ("track_id", "artist_id", "song_hotttnesss"),
        engine="pyarrow",
    ):
        track_id = str(track)
        finite_pop = (
            float(song_popularity)
            if song_popularity is not None and math.isfinite(float(song_popularity))
            else None
        )
        if finite_pop is not None and finite_pop <= tail_threshold:
            catalog_tail_count += 1
        for cutoff, counts in top_counts.items():
            occurrences = counts.get(track_id, 0)
            if not occurrences:
                continue
            accumulator = accumulators[cutoff]
            if artist:
                accumulator["artists"].add(str(artist))
            if finite_pop is not None:
                accumulator["pop_sum"] += occurrences * finite_pop
                accumulator["pop_count"] += occurrences
                if finite_pop <= tail_threshold:
                    accumulator["tail_tracks"].add(track_id)
    return {
        f"top_{cutoff}": {
            "catalog_track_coverage": len(counts) / catalog_tracks,
            "artist_coverage": len(accumulators[cutoff]["artists"]) / len(catalog_artists),
            "average_popularity": (
                accumulators[cutoff]["pop_sum"] / accumulators[cutoff]["pop_count"]
                if accumulators[cutoff]["pop_count"] else None
            ),
            "pop_lift_ratio": (
                accumulators[cutoff]["pop_sum"]
