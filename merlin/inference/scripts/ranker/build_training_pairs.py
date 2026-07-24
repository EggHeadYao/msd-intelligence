"""Construct tuning pairs or streamed final-retrain pairs and features."""

from __future__ import annotations

import argparse
from itertools import islice
import json
import math
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from merlin.embedding.graph.config import GRAPH_CONTRACT_KEY, GRAPH_CONTRACT_VERSION

from ...artifact_paths import InferenceArtifactPaths
from ...candidate_policy import load_candidate_policy
from ...candidate_pool import load_candidate_pool_manifest
from ...catalog_data import load_catalog_context
from ...faiss_index import FaissTrackIndex
from ...features_v2 import PairSignalLookups, RankerV2FeatureComputer
from ...loaders import load_audio_index
from ...recall import RecallPipeline
from ...recall_factory import build_canonical_retrievers
from ...retrieval import TagRetriever, VectorRetriever
from ...scratch import prepare_scratch_root
from ...split import load_split_assignments, load_split_manifest
from ...tag_data import load_tag_idf
from ...training.pairs import construct_query_pairs, write_training_and_feature_artifacts
from ...training.pairs import write_training_pair_artifacts
from ...training.weak_labels import MAX_POSITIVES_PER_QUERY, WEAK_LABEL_VERSION
from ...training.weak_labels import load_weak_positive_manifest, select_weak_positives
from ...types import Candidate


FINAL_SPLITS = frozenset({"set_a", "set_b", "remaining"})


def _load_thresholds(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        thresholds = json.load(stream)
    if thresholds.get("artifact_type") != "weak_label_thresholds":
        raise ValueError("weak-label threshold artifact type mismatch")
    if thresholds.get("artifact_version") != WEAK_LABEL_VERSION:
        raise ValueError("weak-label threshold artifact version mismatch")
    if thresholds.get("fit_split") != "set_a":
        raise ValueError("weak-label thresholds must be frozen from Set A")
    return thresholds


def _tag_neighbors(tag: TagRetriever, query_id: str) -> list[tuple[str, float]]:
    artist = tag.track_to_artist.get(query_id)
    if artist is None:
        return []
    artists = (
        tag.similar_artists(artist)
        if callable(tag.similar_artists)
        else tag.similar_artists.get(artist, ())
    )
    return [
        (track_id, float(score))
        for target_artist, score in artists
        for track_id in sorted(tag.artist_tracks.get(target_artist, ()))
    ]


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_predicates(audio, tag: TagRetriever, thresholds: Mapping[str, object]):
    audio_threshold = float(thresholds["audio_cosine_p90"])
    tag_threshold = float(thresholds["tag_tfidf_cosine_p90"])
    positive_cache: dict[tuple[str, str], bool] = {}
    cached_query_id: str | None = None

    def is_positive(query_id: str, candidate_id: str) -> bool:
        nonlocal cached_query_id
        if query_id != cached_query_id:
            positive_cache.clear()
            cached_query_id = query_id
        key = (query_id, candidate_id)
        cached = positive_cache.get(key)
        if cached is not None:
            return cached
        query_artist = tag.track_to_artist.get(query_id)
        candidate_artist = tag.track_to_artist.get(candidate_id)
        if query_artist is not None and query_artist == candidate_artist:
            result = True
        elif query_artist is None or candidate_artist is None:
            result = False
        else:
            audio_score = audio.similarity(query_id, candidate_id)
            if audio_score is not None and audio_score >= audio_threshold:
                result = True
            else:
                tag_score = tag.pair_score(query_id, candidate_id)
                result = tag_score is not None and tag_score >= tag_threshold
        positive_cache[key] = result
        return result

    def is_positive_batch(query_id: str, candidate_ids: Sequence[str]) -> list[bool]:
        query_artist = tag.track_to_artist.get(query_id)
        audio_scores = audio.similarities(query_id, candidate_ids)
        results = []
        for candidate_id, audio_score in zip(candidate_ids, audio_scores, strict=True):
            candidate_artist = tag.track_to_artist.get(candidate_id)
            if query_artist is not None and query_artist == candidate_artist:
                results.append(True)
            elif query_artist is None or candidate_artist is None:
                results.append(False)
            elif audio_score is not None and audio_score >= audio_threshold:
                results.append(True)
            else:
                tag_score = tag.pair_score(query_id, candidate_id)
                results.append(tag_score is not None and tag_score >= tag_threshold)
        return results

    return is_positive, is_positive_batch


def main() -> None:
    args = parse_args()
    paths = InferenceArtifactPaths()
    if args.stage == "tuning":
        _run_tuning(args, paths)
    else:
        _run_final(args, paths)


def parse_args() -> argparse.Namespace:
    defaults = InferenceArtifactPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, default=defaults.candidate_pool)
    parser.add_argument(
        "--candidate-pool-manifest",
        type=Path,
        default=defaults.candidate_pool_manifest,
    )
    parser.add_argument("--weak-positives", type=Path, default=defaults.weak_positives)
    parser.add_argument(
        "--weak-positives-manifest",
        type=Path,
        default=defaults.weak_positives_manifest,
    )
    parser.add_argument("--thresholds", type=Path, default=defaults.weak_label_thresholds)
    parser.add_argument("--split-assignments", type=Path, default=defaults.split_assignments)
    parser.add_argument("--split-manifest", type=Path, default=defaults.split_manifest)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--features-output", type=Path)
    parser.add_argument("--features-manifest", type=Path)
    parser.add_argument("--stage", choices=("tuning", "final_retrain"), default="tuning")
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rows-per-file", type=int, default=250_000)
    parser.add_argument("--positive-neighbor-limit", type=int, default=1_001)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--graph-contract-key", default=GRAPH_CONTRACT_KEY)
    parser.add_argument("--graph-contract-version", default=GRAPH_CONTRACT_VERSION)
    parser.add_argument("--min-free-gb", type=float)
    return parser.parse_args()


def _run_tuning(args: argparse.Namespace, paths: InferenceArtifactPaths) -> None:
    output = args.output or paths.training_pairs
    manifest_path = args.manifest or paths.training_pairs_manifest
    if args.features_output is not None or args.features_manifest is not None:
        raise ValueError("tuning features are exported by export_ranker_features")
    load_candidate_pool_manifest(
        args.candidate_pool_manifest,
        args.candidate_pool,
        expected_scope=args.scope,
    )
    weak_manifest = load_weak_positive_manifest(
        args.weak_positives_manifest,
        args.weak_positives,
        args.thresholds,
        expected_scope=args.scope,
    )
    assignments = load_split_assignments(args.split_assignments)
    positive_count = int(weak_manifest.get("positive_count", 0))
    if positive_count <= 0:
        raise ValueError("weak-positive manifest has no positive rows")
    prepare_scratch_root(
        output.parent,
        scope=args.scope,
        min_free_gb=args.min_free_gb,
        projected_gb=positive_count * 4 * 80 / (1024**3),
    )
    thresholds = _load_thresholds(args.thresholds)
    audio = load_audio_index()
    catalog = load_catalog_context(paths.songs_metadata, paths.graph_edges)
    same_song = catalog.same_song
    tag = TagRetriever.from_data(
        catalog.tag_data,
        idf_values=load_tag_idf(
            paths.tag_idf,
            expected_graph_edges_path=paths.graph_edges,
        ),
        same_song=same_song,
    )

    is_positive, is_positive_batch = _positive_predicates(audio, tag, thresholds)

    manifest = write_training_pair_artifacts(
        args.candidate_pool,
        args.weak_positives,
        assignments,
        output,
        manifest_path,
        stage="tuning",
        same_song=same_song,
        is_positive=is_positive,
        is_positive_batch=is_positive_batch,
        parent_paths={
            "candidate_pool": args.candidate_pool,
            "weak_positives": args.weak_positives,
            "weak_label_thresholds": args.thresholds,
            "split_assignments": args.split_assignments,
            "audio_index_manifest": paths.audio_manifest,
            "tag_idf": paths.tag_idf,
        },
        scope=args.scope,
    )
    print(
        "training_pairs_ready "
        f"scope={args.scope} stage={args.stage} pairs={manifest['pair_count']} "
        f"output={output}",
    )


def _select_final_positives(
    query_id: str,
    allowed: set[str],
    neighbors: Sequence[tuple[str, float]],
    audio_retriever: VectorRetriever,
    tag: TagRetriever,
    thresholds: Mapping[str, object],
) -> tuple[str | None, dict[str, frozenset[str]]]:
    artist = tag.track_to_artist.get(query_id)
    selected = select_weak_positives(
        query_id,
        allowed,
        tag.track_to_artist,
        tag.artist_tracks.get(artist, ()) if artist else (),
        neighbors,
        _tag_neighbors(tag, query_id),
        audio_retriever.same_song,
        thresholds,
        limit=MAX_POSITIVES_PER_QUERY,
    )
    positives = {
        str(row["track_id"]): frozenset(
            str(source) for source in row["positive_sources"]
        )
        for row in selected
    }
    return artist, positives


def _final_positive_checks(
    query_id: str,
    artist: str | None,
    audio_cache: dict[str, float | None],
    audio,
    tag: TagRetriever,
    computer: RankerV2FeatureComputer,
    thresholds: Mapping[str, object],
):
    audio_threshold = float(thresholds["audio_cosine_p90"])
    tag_threshold = float(thresholds["tag_tfidf_cosine_p90"])

    def audio_value(candidate_id: str) -> float | None:
        if candidate_id not in audio_cache:
            audio_cache[candidate_id] = _finite(
                computer.signals.audio(query_id, candidate_id)
            )
        return audio_cache[candidate_id]

    def is_positive(_query: str, candidate_id: str) -> bool:
        candidate_artist = tag.track_to_artist.get(candidate_id)
        if artist is not None and artist == candidate_artist:
            return True
        if artist is None or candidate_artist is None:
            return False
        audio_score = audio_value(candidate_id)
        if audio_score is not None and audio_score >= audio_threshold:
            return True
        tag_score = _finite(tag.pair_score(query_id, candidate_id))
        return tag_score is not None and tag_score >= tag_threshold

    def is_positive_batch(_query: str, candidate_ids: Sequence[str]) -> list[bool]:
        missing = [track_id for track_id in candidate_ids if track_id not in audio_cache]
        if missing:
            scores = audio.similarities(query_id, missing)
            audio_cache.update(
                (track_id, _finite(score))
                for track_id, score in zip(missing, scores, strict=True)
            )
        return [is_positive(query_id, track_id) for track_id in candidate_ids]

    return is_positive, is_positive_batch


def _final_feature_rows(
    query_id: str,
    rows: Sequence[Mapping[str, object]],
    candidates: Sequence[Candidate],
    audio_cache: Mapping[str, float | None],
    computer: RankerV2FeatureComputer,
) -> list[dict[str, object]]:
    recalled_by_id = {candidate.track_id: candidate for candidate in candidates}
    feature_candidates = []
    for row in rows:
        candidate_id = str(row["candidate_track_id"])
        recalled_candidate = recalled_by_id.get(candidate_id)
        scores = dict(recalled_candidate.recall_scores) if recalled_candidate else {}
        audio_score = audio_cache.get(candidate_id)
        if audio_score is not None:
            scores["audio"] = audio_score
        feature_candidates.append(Candidate(candidate_id, recall_scores=scores))
    raw_rows = computer.compute_raw_many(query_id, feature_candidates)
    return [
        {
            "query_track_id": query_id,
            "candidate_track_id": candidate.track_id,
            "label": int(row["label"]),
            **raw,
        }
        for row, candidate, raw in zip(rows, feature_candidates, raw_rows, strict=True)
    ]


def _final_query_rows(
    queries: Sequence[str],
    allowed: set[str],
    thresholds: Mapping[str, object],
    pipeline: RecallPipeline,
    audio,
    audio_retriever: VectorRetriever,
    tag: TagRetriever,
    computer: RankerV2FeatureComputer,
    batch_size: int,
    positive_neighbor_limit: int,
) -> Iterator[tuple[list[dict[str, object]], list[dict[str, object]], Mapping[str, object]]]:
    universe = tuple(sorted(allowed))
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        audio_neighbors = audio.search_many(batch, positive_neighbor_limit)
        audio_override = {
            query_id: audio_retriever.filter_neighbors(
                query_id, neighbors, pipeline.retriever_limits["audio"]
            )
            for query_id, neighbors in zip(batch, audio_neighbors, strict=True)
        }
        recalled = pipeline.recall_many(batch, source_overrides={"audio": audio_override})
        for query_id, neighbors in zip(batch, audio_neighbors, strict=True):
            candidates, _audit = recalled[query_id]
            artist, positives = _select_final_positives(
                query_id, allowed, neighbors, audio_retriever, tag, thresholds
            )
            audio_cache = {
                track_id: _finite(score)
                for track_id, score in neighbors
                if track_id in allowed
            }
            checks = _final_positive_checks(
                query_id, artist, audio_cache, audio, tag, computer, thresholds
            )
            rows, audit = construct_query_pairs(
                query_id,
                positives,
                candidates,
                allowed,
                universe,
                audio_retriever.same_song,
                *checks,
            )
            if rows:
                features = _final_feature_rows(
                    query_id, rows, candidates, audio_cache, computer
                )
                yield rows, features, audit
        processed = min(start + len(batch), len(queries))
        print(f"final_retrain_progress queries={processed}/{len(queries)}", flush=True)


def _final_run_config(args: argparse.Namespace, paths: InferenceArtifactPaths):
    sizes = (args.batch_size, args.rows_per_file, args.positive_neighbor_limit)
    if any(value <= 0 for value in sizes):
        raise ValueError("final-retrain batch, part, and neighbor sizes must be positive")
    if args.limit_queries < 0:
        raise ValueError("limit-queries must be non-negative")
    load_split_manifest(args.split_manifest, args.split_assignments)
    assignments = load_split_assignments(args.split_assignments)
    allowed = {
        track_id for track_id, split in assignments.items() if split in FINAL_SPLITS
    }
    queries = tuple(sorted(allowed))
    if args.limit_queries:
        queries = tuple(islice(queries, args.limit_queries))
    scope = "smoke" if args.limit_queries else args.scope
    output = args.output or paths.final_training_pairs
    manifest_path = args.manifest or paths.final_training_pairs_manifest
    feature_output = args.features_output or paths.final_raw_features
    feature_manifest = args.features_manifest or paths.final_raw_features_manifest
    projected_gb = len(queries) * MAX_POSITIVES_PER_QUERY * 4 * 48 / (1024**3)
    prepare_scratch_root(
        output.parent,
        scope=scope,
        min_free_gb=args.min_free_gb,
        projected_gb=projected_gb,
    )
    return (
        allowed,
        queries,
        scope,
        output,
        manifest_path,
        feature_output,
        feature_manifest,
    )


def _final_runtime(args: argparse.Namespace, paths: InferenceArtifactPaths):
    thresholds = _load_thresholds(args.thresholds)
    audio = load_audio_index()
    graph = FaissTrackIndex.from_files(
        paths.graph_index,
        paths.graph_mapping,
        paths.graph_manifest,
        paths.graph_encoder_metadata,
        expected_space="graph",
        expected_contract_key=args.graph_contract_key,
        expected_contract=args.graph_contract_version,
    )
    catalog = load_catalog_context(
        paths.songs_metadata,
        paths.graph_edges,
        include_ranker_metadata=True,
    )
    tag = TagRetriever.from_data(
        catalog.tag_data,
        idf_values=load_tag_idf(
            paths.tag_idf,
            expected_graph_edges_path=paths.graph_edges,
        ),
        same_song=catalog.same_song,
    )
    retrievers = build_canonical_retrievers(
        audio, graph, paths, catalog.same_song, tag
    )
    policy = load_candidate_policy(paths.candidate_policy)
    pipeline = RecallPipeline(
        retrievers=retrievers,
        retriever_limits={
            str(name): int(limit)
            for name, limit in policy["retriever_limits"].items()
        },
        candidate_limit=int(policy["candidate_limit"]),
        canonical=True,
    )
    audio_retriever = next(
        retriever
        for retriever in retrievers
        if isinstance(retriever, VectorRetriever) and retriever.name == "audio"
    )
    _audio, _graph, bfs, tag = retrievers
    computer = RankerV2FeatureComputer(
        tracks=catalog.ranker_tracks,
        signals=PairSignalLookups(
            audio=audio.similarity,
            graph=graph.similarity,
            bfs=bfs.pair_score,
            tags=tag.pair_score,
            audio_batch=audio.similarities,
            graph_batch=graph.similarities,
        ),
    )
    return thresholds, audio, audio_retriever, tag, computer, pipeline


def _run_final(args: argparse.Namespace, paths: InferenceArtifactPaths) -> None:
    (
        allowed,
        queries,
        scope,
        output,
        manifest_path,
        feature_output,
        feature_manifest,
    ) = _final_run_config(args, paths)
    thresholds, audio, audio_retriever, tag, computer, pipeline = _final_runtime(
        args, paths
    )
    pair_manifest, _feature_manifest = write_training_and_feature_artifacts(
        _final_query_rows(
            queries,
            allowed,
            thresholds,
            pipeline,
            audio,
            audio_retriever,
            tag,
            computer,
            args.batch_size,
            args.positive_neighbor_limit,
        ),
        output,
        manifest_path,
        feature_output,
        feature_manifest,
        parent_paths={
            "split_manifest": args.split_manifest,
            "split_assignments": args.split_assignments,
            "weak_label_thresholds": args.thresholds,
            "audio_index_manifest": paths.audio_manifest,
            "graph_index_manifest": paths.graph_manifest,
            "candidate_policy_manifest": paths.candidate_policy,
            "tag_idf": paths.tag_idf,
            "songs_metadata": paths.songs_metadata,
            "graph_edges": paths.graph_edges,
        },
        scope=scope,
        rows_per_file=args.rows_per_file,
    )
    print(
        f"training_pairs_ready scope={scope} stage=final_retrain "
        f"pairs={pair_manifest['pair_count']} output={output} "
        f"features={feature_output}"
    )


if __name__ == "__main__":
    main()
