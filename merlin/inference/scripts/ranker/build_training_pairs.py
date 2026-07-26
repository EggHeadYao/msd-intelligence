"""Construct tuning pairs or streamed retrain pairs and features."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
import math
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from merlin.embedding.graph.config import GRAPH_CONTRACT_KEY, GRAPH_CONTRACT_VERSION

from ...artifacts.integrity import sha256_path
from ...artifacts.paths import InferenceArtifactPaths
from ...recall.policy import load_candidate_policy
from ...recall.pool import load_candidate_pool_manifest
from ...data.catalog import load_catalog_context
from ...recall.streaming import EncodedCandidates, StreamingRecallEngine, TrackCodec
from ...retrieval.faiss import FaissTrackIndex
from ...artifacts.io import PartitionedParquetWriter, write_json_atomic
from ...retrieval.faiss import load_audio_index
from ...recall.factory import build_canonical_retrievers
from ...retrieval import TagRetriever, VectorRetriever
from ..support.scratch import prepare_scratch_root
from ...training.split import load_split_assignments, load_split_manifest
from ...data.tags import load_tag_idf
from ...training.pairs import (
    CANDIDATE_AWARE_FRACTION,
    finish_query_pairs,
    prepare_query_pairs,
    StreamCheckpoint,
    StreamTableBatch,
    construct_query_pairs,
    load_stream_checkpoint,
    load_training_pair_manifest,
    sample_random_negatives_many,
    sample_random_negatives_many_by_query,
    training_pair_parquet_schema,
    write_training_manifests_from_stats,
    write_training_and_feature_artifacts,
)
from ...ranking.features import (
    PairSignalLookups,
    RAW_BASE_FEATURES,
    RankerFeatureComputer,
    SAMPLE_WEIGHT_COLUMN,
    load_raw_feature_manifest,
    raw_feature_parquet_schema,
)
from ...training.pairs import write_training_pair_artifacts
from ...training.weak_labels import MAX_POSITIVES_PER_QUERY, WEAK_LABEL_VERSION
from ...training.weak_labels import load_weak_positive_manifest, select_weak_positives
from ...types import Candidate


SPLITS = frozenset({"set_a", "set_b", "remaining"})
FEATURE_PAIR_BATCH_SIZE = 32_768
CHECKPOINT_QUERY_INTERVAL = 1_024


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
    if args.stage == "tuning" and args.negative_mode != "candidate_aware":
        raise ValueError("random-only negatives are defined for retrain only")
    if args.stage == "tuning":
        _run_tuning(args, paths)
    else:
        _run(args, paths)


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
    parser.add_argument(
        "--full-training-pairs",
        type=Path,
        default=defaults.training_pairs,
    )
    parser.add_argument(
        "--full-training-pairs-manifest",
        type=Path,
        default=defaults.training_pairs_manifest,
    )
    parser.add_argument(
        "--full-features",
        type=Path,
        default=defaults.raw_pair_features,
    )
    parser.add_argument(
        "--full-features-manifest",
        type=Path,
        default=defaults.raw_pair_features_manifest,
    )
    parser.add_argument("--stage", choices=("tuning", "final_retrain"), default="tuning")
    parser.add_argument(
        "--negative-mode",
        choices=("candidate_aware", "random_only"),
        default="candidate_aware",
    )
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
    output = args.output or paths.tuning_training_pairs
    manifest_path = args.manifest or paths.tuning_training_pairs_manifest
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


def _select_positives(
    query_id: str,
    allowed: set[str],
    neighbors: Sequence[tuple[str, float]],
    tag_neighbors: Sequence[tuple[str, float]],
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
        tag_neighbors,
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


def _positive_checks(
    query_id: str,
    artist: str | None,
    audio_cache: dict[str, float | None],
    audio,
    tag: TagRetriever,
    computer: RankerFeatureComputer,
    thresholds: Mapping[str, object],
):
    audio_threshold = float(thresholds["audio_cosine_p90"])
    tag_threshold = float(thresholds["tag_tfidf_cosine_p90"])
    tag_cache: dict[str, float | None] = {}
    tag_pair_cache: dict[str, float | None] = {}

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
        results: list[bool] = []
        for candidate_id in candidate_ids:
            candidate_artist = tag.track_to_artist.get(candidate_id)
            if artist is not None and artist == candidate_artist:
                results.append(True)
                continue
            if artist is None or candidate_artist is None:
                results.append(False)
                continue
            audio_score = audio_cache[candidate_id]
            if audio_score is not None and audio_score >= audio_threshold:
                results.append(True)
                continue
            if candidate_artist not in tag_cache:
                tag_cache[candidate_artist] = _finite(
                    tag.pair_score(query_id, candidate_id)
                )
            tag_score = tag_cache[candidate_artist]
            tag_pair_cache[candidate_id] = tag_score
            results.append(tag_score is not None and tag_score >= tag_threshold)
        return results

    return is_positive, is_positive_batch, {
        "audio": audio_cache,
        "tag": tag_pair_cache,
    }


def _table_batch(
    states: Sequence[
        tuple[
            str,
            Sequence[Mapping[str, object]],
            Sequence[Candidate] | EncodedCandidates,
            Mapping[str, Mapping[str, float | None]],
        ]
    ],
    computer: RankerFeatureComputer,
    audits: Sequence[Mapping[str, object]],
) -> StreamTableBatch:
    """Materialize one aligned pair/feature batch directly as Arrow columns."""
    import pyarrow as pa

    pair_inputs: list[tuple[str, str, Mapping[str, float]]] = []
    pair_rows: list[Mapping[str, object]] = []
    weak_sources: Counter[str] = Counter()
    recall_sources: Counter[str] = Counter()
    for query_id, rows, candidates, signal_caches in states:
        recalled_by_id = (
            {candidate.track_id: candidate for candidate in candidates}
            if not isinstance(candidates, EncodedCandidates)
            else None
        )
        for row in rows:
            pair_rows.append(row)
            weak_sources.update(row["positive_sources"])
            recall_sources.update(row["recall_sources"])
            candidate_id = str(row["candidate_track_id"])
            if isinstance(candidates, EncodedCandidates):
                candidate_code = candidates.codec.code(candidate_id)
                candidate_position = candidates.position(candidate_code)
                scores = (
                    candidates.evidence(candidate_position)[1]
                    if candidate_position is not None
                    else {}
                )
            else:
                assert recalled_by_id is not None
                recalled = recalled_by_id.get(candidate_id)
                scores = dict(recalled.recall_scores) if recalled else {}
            for source, cache in signal_caches.items():
                score = cache.get(candidate_id)
                if score is not None and (
                    source == "audio" or source not in scores
                ):
                    scores[source] = score
            pair_inputs.append((query_id, candidate_id, scores))
    raw_columns = {name: [] for name in RAW_BASE_FEATURES}
    for start in range(0, len(pair_inputs), FEATURE_PAIR_BATCH_SIZE):
        columns = computer.compute_raw_pair_columns(
            pair_inputs[start : start + FEATURE_PAIR_BATCH_SIZE]
        )
        for name in RAW_BASE_FEATURES:
            raw_columns[name].extend(columns[name])
    if any(len(pair_rows) != len(values) for values in raw_columns.values()):
        raise ValueError("pair and raw-feature batch counts differ")
    pair_columns = {
        "query_track_id": [str(row["query_track_id"]) for row in pair_rows],
        "candidate_track_id": [str(row["candidate_track_id"]) for row in pair_rows],
        "label": [int(row["label"]) for row in pair_rows],
        SAMPLE_WEIGHT_COLUMN: [
            float(row[SAMPLE_WEIGHT_COLUMN]) for row in pair_rows
        ],
        "positive_sources": [row["positive_sources"] for row in pair_rows],
        "negative_source": [row["negative_source"] for row in pair_rows],
        "recall_sources": [row["recall_sources"] for row in pair_rows],
    }
    feature_columns = {
        name: pair_columns[name]
        for name in (
            "query_track_id",
            "candidate_track_id",
            "label",
            SAMPLE_WEIGHT_COLUMN,
        )
    }
    feature_columns.update(raw_columns)
    return StreamTableBatch(
        pa.Table.from_pydict(pair_columns, schema=training_pair_parquet_schema()),
        pa.Table.from_pydict(
            feature_columns,
            schema=raw_feature_parquet_schema("training"),
        ),
        tuple(audits),
        weak_sources,
        recall_sources,
    )


def _query_rows(
    queries: Sequence[str],
    allowed: set[str],
    thresholds: Mapping[str, object],
    recall_engine: StreamingRecallEngine,
    audio,
    audio_retriever: VectorRetriever,
    tag: TagRetriever,
    computer: RankerFeatureComputer,
    batch_size: int,
    positive_neighbor_limit: int,
    *,
    query_offset: int = 0,
    total_query_count: int | None = None,
) -> Iterator[StreamTableBatch | StreamCheckpoint]:
    total = len(queries) if total_query_count is None else total_query_count
    universe = tuple(sorted(allowed))
    last_checkpoint = query_offset
    indexed_batches = iter(
        (start, queries[start : start + batch_size])
        for start in range(0, len(queries), batch_size)
    )
    current = next(indexed_batches, None)
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        recalled_job = (
            prefetch.submit(
                recall_engine.search_many, current[1], positive_neighbor_limit
            )
            if current is not None
            else None
        )
        while current is not None and recalled_job is not None:
            start, batch = current
            recalled = recalled_job.result()
            following = next(indexed_batches, None)
            following_job = (
                prefetch.submit(
                    recall_engine.search_many,
                    following[1],
                    positive_neighbor_limit,
                )
                if following is not None
                else None
            )
            prepared_states = []
            for position, query_id in enumerate(batch):
                candidates, neighbors, tag_neighbors = recall_engine.query(
                    recalled,
                    position,
                )
                artist, positives = _select_positives(
                    query_id,
                    allowed,
                    neighbors,
                    tag_neighbors,
                    audio_retriever,
                    tag,
                    thresholds,
                )
                audio_cache = {
                    track_id: _finite(score)
                    for track_id, score in neighbors
                    if track_id in allowed
                }
                is_positive, is_positive_batch, signal_caches = _positive_checks(
                    query_id,
                    artist,
                    audio_cache,
                    audio,
                    tag,
                    computer,
                    thresholds,
                )
                prepared = prepare_query_pairs(
                    query_id,
                    positives,
                    candidates,
                    allowed,
                    audio_retriever.same_song,
                    is_positive,
                    is_positive_batch,
                )
                if prepared is not None:
                    prepared_states.append((prepared, candidates, signal_caches))
            requests = [
                (
                    prepared.query_id,
                    prepared.negative_target - len(prepared.candidate_selected),
                    set(prepared.selected_positives)
                    | {
                        track_id
                        for track_id, _evidence in prepared.candidate_selected
                    },
                )
                for prepared, _candidates, _signal_caches in prepared_states
            ]
            audio_caches = {
                prepared.query_id: signal_caches["audio"]
                for prepared, _candidates, signal_caches in prepared_states
            }
            tag_caches = {
                prepared.query_id: signal_caches["tag"]
                for prepared, _candidates, signal_caches in prepared_states
            }
            random_by_query, random_rejections = (
                sample_random_negatives_many_by_query(
                    requests,
                    universe,
                    audio_retriever.same_song,
                    lambda pairs: _derived_positive_pair_flags(
                        pairs,
                        audio,
                        tag,
                        thresholds,
                        audio_caches,
                        tag_caches,
                    ),
                )
            )
            states = []
            audits = []
            for prepared, candidates, signal_caches in prepared_states:
                rows, audit = finish_query_pairs(
                    prepared,
                    random_by_query[prepared.query_id],
                    random_rejections[prepared.query_id],
                )
                states.append((prepared.query_id, rows, candidates, signal_caches))
                audits.append(audit)
            if states:
                yield _table_batch(states, computer, audits)
            processed = query_offset + min(start + len(batch), len(queries))
            if (
                processed == total
                or processed - last_checkpoint >= CHECKPOINT_QUERY_INTERVAL
            ):
                yield StreamCheckpoint(processed, total)
                last_checkpoint = processed
            print(
                f"retrain_progress queries={processed}/{total}",
                flush=True,
            )
            current = following
            recalled_job = following_job


def _run_config(args: argparse.Namespace, paths: InferenceArtifactPaths):
    sizes = (args.batch_size, args.rows_per_file, args.positive_neighbor_limit)
    if any(value <= 0 for value in sizes):
        raise ValueError("retrain batch, part, and neighbor sizes must be positive")
    if args.limit_queries < 0:
        raise ValueError("limit-queries must be non-negative")
    load_split_manifest(args.split_manifest, args.split_assignments)
    assignments = load_split_assignments(args.split_assignments)
    allowed = {
        track_id for track_id, split in assignments.items() if split in SPLITS
    }
    queries = tuple(sorted(allowed))
    if args.limit_queries:
        queries = tuple(islice(queries, args.limit_queries))
    scope = "smoke" if args.limit_queries else args.scope
    random_only = args.negative_mode == "random_only"
    output = args.output or (
        paths.no_hard_neg_pairs if random_only else paths.training_pairs
    )
    manifest_path = args.manifest or (
        paths.no_hard_neg_pairs_manifest
        if random_only
        else paths.training_pairs_manifest
    )
    feature_output = args.features_output or (
        paths.no_hard_neg_raw_features if random_only else paths.raw_pair_features
    )
    feature_manifest = args.features_manifest or (
        paths.no_hard_neg_raw_features_manifest
        if random_only
        else paths.raw_pair_features_manifest
    )
    projected_gb = len(queries) * MAX_POSITIVES_PER_QUERY * 4 * 48 / (1024**3)
    prepare_scratch_root(
        output.parent,
        scope=scope,
        min_free_gb=args.min_free_gb,
        projected_gb=projected_gb,
    )
    return (
        assignments,
        allowed,
        queries,
        scope,
        output,
        manifest_path,
        feature_output,
        feature_manifest,
    )


def _runtime(
    args: argparse.Namespace,
    paths: InferenceArtifactPaths,
    assignments: Mapping[str, str],
    *,
    include_recall: bool = True,
):
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
    audio_retriever = next(
        retriever
        for retriever in retrievers
        if isinstance(retriever, VectorRetriever) and retriever.name == "audio"
    )
    _audio, _graph, bfs, tag = retrievers
    recall_engine = None
    if include_recall:
        policy = load_candidate_policy(paths.candidate_policy)
        limits = {
            str(name): int(limit)
            for name, limit in policy["retriever_limits"].items()
        }
        recall_engine = StreamingRecallEngine(
            audio,
            graph,
            bfs,
            tag,
            TrackCodec.build(assignments, SPLITS, catalog.same_song),
            limits,
        )
    computer = RankerFeatureComputer(
        tracks=catalog.ranker_tracks,
        signals=PairSignalLookups(
            audio=audio.similarity,
            graph=graph.similarity,
            bfs=bfs.pair_score,
            tags=tag.pair_score,
            audio_batch=audio.similarities,
            graph_batch=graph.similarities,
            bfs_batch=lambda query_id, candidate_ids: bfs.pair_scores(
                [(query_id, candidate_id) for candidate_id in candidate_ids]
            ),
            tags_batch=lambda query_id, candidate_ids: tag.pair_scores(
                [(query_id, candidate_id) for candidate_id in candidate_ids]
            ),
            audio_pairs=audio.pair_similarities,
            graph_pairs=graph.pair_similarities,
            bfs_pairs=bfs.pair_scores,
            tags_pairs=tag.pair_scores,
        ),
    )
    return thresholds, audio, audio_retriever, tag, computer, recall_engine


def _parquet_parts(path: Path) -> tuple[Path, ...]:
    parts = (path,) if path.is_file() else tuple(sorted(path.glob("part-*.parquet")))
    if not parts:
        raise ValueError(f"Parquet artifact contains no data files: {path}")
    return parts


def _trim_derived_parts(path: Path, keep: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not temporary.is_dir():
        raise FileNotFoundError(f"derived checkpoint dataset is missing: {temporary}")
    for part in temporary.glob("part-*.parquet"):
        if int(part.stem.split("-")[-1]) >= keep:
            part.unlink()
    (temporary / "_SUCCESS").unlink(missing_ok=True)


def _load_derivation_checkpoint(
    path: Path,
    contract: Mapping[str, object],
    pair_output: Path,
    feature_output: Path,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    if (
        checkpoint.get("artifact_type") != "no_hard_neg_derivation_checkpoint"
        or checkpoint.get("artifact_version") != 2
        or checkpoint.get("contract") != dict(contract)
    ):
        raise ValueError("no-hard-neg derivation checkpoint contract mismatch")
    pair_parts = int(checkpoint.get("pair_part_count", -1))
    feature_parts = int(checkpoint.get("feature_part_count", -1))
    if pair_parts < 0 or feature_parts < 0:
        raise ValueError("no-hard-neg checkpoint part counts are invalid")
    _trim_derived_parts(pair_output, pair_parts)
    _trim_derived_parts(feature_output, feature_parts)
    return checkpoint


def _derived_positive_pair_flags(
    pairs: Sequence[tuple[str, str]],
    audio,
    tag: TagRetriever,
    thresholds: Mapping[str, object],
    audio_caches: Mapping[str, dict[str, float | None]],
    tag_caches: Mapping[str, dict[str, float | None]] | None = None,
) -> list[bool]:
    """Batch weak-positive exclusion checks across independent queries."""
    results = [False] * len(pairs)
    unresolved_indexes = []
    unresolved_pairs = []
    for index, (query_id, candidate_id) in enumerate(pairs):
        query_artist = tag.track_to_artist.get(query_id)
        candidate_artist = tag.track_to_artist.get(candidate_id)
        if query_artist is not None and query_artist == candidate_artist:
            results[index] = True
        elif query_artist is not None and candidate_artist is not None:
            unresolved_indexes.append(index)
            unresolved_pairs.append((query_id, candidate_id))
    audio_scores: list[float | None] = [None] * len(unresolved_pairs)
    positions_by_query: dict[str, list[int]] = {}
    for position, (query_id, _candidate_id) in enumerate(unresolved_pairs):
        positions_by_query.setdefault(query_id, []).append(position)
    for query_id, positions in positions_by_query.items():
        candidate_ids = [unresolved_pairs[position][1] for position in positions]
        scores = audio.similarities(query_id, candidate_ids)
        for position, score in zip(positions, scores, strict=True):
            audio_scores[position] = score
    tag_indexes = []
    tag_pairs = []
    audio_threshold = float(thresholds["audio_cosine_p90"])
    for index, pair, score in zip(
        unresolved_indexes, unresolved_pairs, audio_scores, strict=True
    ):
        value = _finite(score)
        audio_caches[pair[0]][pair[1]] = value
        if value is not None and value >= audio_threshold:
            results[index] = True
        else:
            tag_indexes.append(index)
            tag_pairs.append(pair)
    tag_threshold = float(thresholds["tag_tfidf_cosine_p90"])
    tag_pair_lookup = getattr(tag, "pair_scores", None)
    tag_scores = (
        tag_pair_lookup(tag_pairs)
        if tag_pair_lookup is not None
        else [tag.pair_score(*pair) for pair in tag_pairs]
    )
    for index, pair, score in zip(
        tag_indexes,
        tag_pairs,
        tag_scores,
        strict=True,
    ):
        value = _finite(score)
        if tag_caches is not None:
            tag_caches[pair[0]][pair[1]] = value
        results[index] = value is not None and value >= tag_threshold
    return results


def _run_random_only_derived(
    args: argparse.Namespace,
    paths: InferenceArtifactPaths,
) -> None:
    """Replace only Full hard negatives while reusing every other row."""
    if args.limit_queries:
        raise ValueError("derived formal no-hard-neg does not support query limits")
    full_pairs = load_training_pair_manifest(
        args.full_training_pairs_manifest,
        args.full_training_pairs,
        expected_scope=args.scope,
        expected_stage="final_retrain",
    )
    full_features = load_raw_feature_manifest(
        args.full_features_manifest,
        args.full_features,
        expected_scope=args.scope,
        expected_pair_kind="training",
        expected_stage="final_retrain",
    )
    if full_features.get("parent_hashes", {}).get(
        "training_pairs_manifest"
    ) != sha256_path(args.full_training_pairs_manifest):
        raise ValueError("Full features are not bound to the Full pair manifest")
    full_counts = full_pairs.get("counts", {})
    query_count = int(full_pairs.get("query_count", 0))
    pair_count = int(full_pairs.get("pair_count", 0))
    if (
        query_count <= 0
        or pair_count <= 0
        or pair_count != int(full_features.get("row_count", 0))
        or pair_count % query_count
    ):
        raise ValueError("Full pair/feature row layout cannot be derived safely")
    rows_per_query = pair_count // query_count
    if (
        int(full_counts.get("positive_count", -1)) * 4 != pair_count
        or int(full_counts.get("negative_count", -1)) * 4 != 3 * pair_count
    ):
        raise ValueError("Full artifact does not preserve the frozen 1:3 budget")
    pair_parts = _parquet_parts(args.full_training_pairs)
    feature_parts = _parquet_parts(args.full_features)
    if len(pair_parts) != len(feature_parts) or [part.name for part in pair_parts] != [
        part.name for part in feature_parts
    ]:
        raise ValueError("Full pair and feature Parquet parts are not aligned")

    (
        assignments,
        allowed,
        _queries,
        scope,
        output,
        manifest_path,
        feature_output,
        feature_manifest,
    ) = _run_config(args, paths)
    universe = tuple(sorted(allowed))
    checkpoint_path = output.with_suffix(output.suffix + ".checkpoint.json")
    contract = {
        "scope": scope,
        "source_pair_manifest_sha256": sha256_path(
            args.full_training_pairs_manifest
        ),
        "source_feature_manifest_sha256": sha256_path(
            args.full_features_manifest
        ),
        "source_part_count": len(pair_parts),
        "rows_per_query": rows_per_query,
        "rows_per_file": args.rows_per_file,
        "output": str(output.resolve()),
        "features_output": str(feature_output.resolve()),
    }
    initial = _load_derivation_checkpoint(
        checkpoint_path,
        contract,
        output,
        feature_output,
    )
    start_part = int(initial.get("next_input_part", 0)) if initial else 0
    totals = Counter(initial.get("totals", {})) if initial else Counter()
    rejection_totals = (
        Counter(initial.get("rejection_totals", {})) if initial else Counter()
    )
    processed_queries = int(initial.get("query_count", 0)) if initial else 0
    thresholds, audio, audio_retriever, tag, computer, _recall = _runtime(
        args,
        paths,
        assignments,
        include_recall=False,
    )

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pair_schema = training_pair_parquet_schema()
    feature_schema = raw_feature_parquet_schema("training")
    resume = initial is not None
    with PartitionedParquetWriter(
        output,
        pair_schema,
        rows_per_file=args.rows_per_file,
        resume=resume,
    ) as pair_writer, PartitionedParquetWriter(
        feature_output,
        feature_schema,
        rows_per_file=args.rows_per_file,
        resume=resume,
    ) as feature_writer:
        expected_pair_rows = int(initial.get("pair_count", 0)) if initial else 0
        expected_feature_rows = int(initial.get("feature_count", 0)) if initial else 0
        if (
            pair_writer.count != expected_pair_rows
            or feature_writer.count != expected_feature_rows
        ):
            raise ValueError("derived checkpoint row counts do not match its parts")
        for part_index in range(start_part, len(pair_parts)):
            pair_table = pq.read_table(pair_parts[part_index]).select(pair_schema.names)
            feature_table = pq.read_table(feature_parts[part_index]).select(
                feature_schema.names
            )
            pair_weight_index = pair_table.schema.get_field_index(
                SAMPLE_WEIGHT_COLUMN
            )
            feature_weight_index = feature_table.schema.get_field_index(
                SAMPLE_WEIGHT_COLUMN
            )
            if not pa.types.is_floating(pair_table.schema[pair_weight_index].type):
                raise ValueError("Full pair sample weights are not floating point")
            if not pa.types.is_floating(
                feature_table.schema[feature_weight_index].type
            ):
                raise ValueError("Full feature sample weights are not floating point")
            pair_table = pair_table.set_column(
                pair_weight_index,
                SAMPLE_WEIGHT_COLUMN,
                pc.cast(pair_table[SAMPLE_WEIGHT_COLUMN], pa.float32()),
            )
            feature_table = feature_table.set_column(
                feature_weight_index,
                SAMPLE_WEIGHT_COLUMN,
                pc.cast(feature_table[SAMPLE_WEIGHT_COLUMN], pa.float32()),
            )
            if pair_table.schema != pair_schema or feature_table.schema != feature_schema:
                raise ValueError("Full Parquet schema changed before derivation")
            if pair_table.num_rows != feature_table.num_rows:
                raise ValueError("Full aligned Parquet part row counts differ")
            for name in (
                "query_track_id",
                "candidate_track_id",
                "label",
                SAMPLE_WEIGHT_COLUMN,
            ):
                if not pair_table[name].equals(feature_table[name]):
                    raise ValueError(f"Full pair/feature alignment mismatch: {name}")
            if pair_table.num_rows % rows_per_query:
                raise ValueError("Full Parquet part splits a query group")
            hard_mask = pc.fill_null(
                pc.equal(pair_table["negative_source"], "candidate_aware"),
                False,
            )
            keep_mask = pc.invert(hard_mask)
            reused_pairs = pair_table.filter(keep_mask)
            reused_features = feature_table.filter(keep_mask)
            unit_weights = pa.repeat(
                pa.scalar(1.0, type=pa.float32()), reused_pairs.num_rows
            )
            reused_pairs = reused_pairs.set_column(
                pair_weight_index, SAMPLE_WEIGHT_COLUMN, unit_weights
            )
            reused_features = reused_features.set_column(
                feature_weight_index, SAMPLE_WEIGHT_COLUMN, unit_weights
            )
            labels = pair_table["label"].combine_chunks()
            positive_count = int(pc.sum(labels).as_py())
            hard_count = int(pc.sum(pc.cast(hard_mask, pa.int64())).as_py())
            random_count = pair_table.num_rows - positive_count - hard_count

            query_ids = pair_table["query_track_id"].combine_chunks()
            candidate_ids = pair_table["candidate_track_id"].combine_chunks()
            negative_sources = pair_table["negative_source"].combine_chunks()
            requests = []
            request_rows = []
            audio_caches: dict[str, dict[str, float | None]] = {}
            part_query_count = pair_table.num_rows // rows_per_query
            previous_query = None
            for query_offset in range(0, pair_table.num_rows, rows_per_query):
                query_id = str(query_ids[query_offset].as_py())
                if (
                    query_id == previous_query
                    or query_ids[query_offset + rows_per_query - 1].as_py() != query_id
                ):
                    raise ValueError("Full query rows are not fixed-width and clustered")
                previous_query = query_id
                candidates = candidate_ids.slice(
                    query_offset, rows_per_query
                ).to_pylist()
                sources = negative_sources.slice(
                    query_offset, rows_per_query
                ).to_pylist()
                replacement_count = sum(
                    source == "candidate_aware" for source in sources
                )
                if replacement_count == 0:
                    continue
                rejected = {
                    str(candidate_id)
                    for candidate_id, source in zip(candidates, sources, strict=True)
                    if source != "candidate_aware"
                }
                requests.append((query_id, replacement_count, rejected))
                request_rows.append(query_id)
                audio_caches[query_id] = {}
            selected_by_query, part_rejections = sample_random_negatives_many(
                requests,
                universe,
                audio_retriever.same_song,
                lambda pairs: _derived_positive_pair_flags(
                    pairs, audio, tag, thresholds, audio_caches
                ),
            )
            states = []
            for query_id in request_rows:
                rows = [
                    {
                        "query_track_id": query_id,
                        "candidate_track_id": candidate_id,
                        "label": 0,
                        SAMPLE_WEIGHT_COLUMN: 1.0,
                        "positive_sources": [],
                        "negative_source": "random",
                        "recall_sources": [],
                    }
                    for candidate_id in selected_by_query[query_id]
                ]
                states.append((
                    query_id,
                    rows,
                    (),
                    {"audio": audio_caches[query_id]},
                ))
            replacements = _table_batch(states, computer, ())
            if (
                replacements.pairs.num_rows != hard_count
                or replacements.features.num_rows != hard_count
            ):
                raise ValueError("derived replacement count differs from removed hard negatives")
            pair_writer.write_table(
                pa.concat_tables((reused_pairs, replacements.pairs))
            )
            feature_writer.write_table(
                pa.concat_tables((reused_features, replacements.features))
            )
            totals["positive_count"] += positive_count
            totals["negative_count"] += random_count + hard_count
            totals["random_count"] += random_count + hard_count
            totals["candidate_aware_count"] += 0
            totals["candidate_shortage"] += 0
            rejection_totals.update(part_rejections)
            processed_queries += part_query_count
            pair_writer.checkpoint()
            feature_writer.checkpoint()
            write_json_atomic(
                {
                    "artifact_type": "no_hard_neg_derivation_checkpoint",
                    "artifact_version": 2,
                    "contract": contract,
                    "next_input_part": part_index + 1,
                    "query_count": processed_queries,
                    "pair_count": pair_writer.count,
                    "feature_count": feature_writer.count,
                    "pair_part_count": pair_writer.part_count,
                    "feature_part_count": feature_writer.part_count,
                    "totals": dict(totals),
                    "rejection_totals": dict(rejection_totals),
                },
                checkpoint_path,
            )
            print(
                f"no_hard_derivation_progress queries={processed_queries}/{query_count}",
                flush=True,
            )
    if processed_queries != query_count or pair_writer.count != pair_count:
        raise ValueError("derived no-hard-neg output is incomplete")
    if (
        totals["positive_count"] != int(full_counts["positive_count"])
        or totals["negative_count"] != int(full_counts["negative_count"])
        or totals["candidate_aware_count"] != 0
        or totals["random_count"] != int(full_counts["negative_count"])
    ):
        raise ValueError("derived no-hard-neg budget differs from Full")
    stats = {
        "query_count": processed_queries,
        "pair_count": pair_writer.count,
        "feature_count": feature_writer.count,
        "pair_part_count": pair_writer.part_count,
        "feature_part_count": feature_writer.part_count,
        "totals": totals,
        "loss_weight_totals": Counter({
            "positive": float(totals["positive_count"]),
            "candidate_aware": 0.0,
            "random": float(totals["negative_count"]),
        }),
        "loss_weight_shape_histogram": Counter({"0:0:0": processed_queries}),
        "rejection_totals": rejection_totals,
        "weak_source_totals": Counter(
            full_pairs.get("weak_positive_source_counts", {})
        ),
        "recall_source_totals": Counter(),
    }
    pair_manifest, _feature_manifest = write_training_manifests_from_stats(
        output,
        manifest_path,
        feature_output,
        feature_manifest,
        stats=stats,
        parent_paths={
            "full_training_pairs": args.full_training_pairs,
            "full_training_pairs_manifest": args.full_training_pairs_manifest,
            "full_raw_features": args.full_features,
            "full_raw_features_manifest": args.full_features_manifest,
            "split_manifest": args.split_manifest,
            "split_assignments": args.split_assignments,
            "weak_label_thresholds": args.thresholds,
            "audio_index_manifest": paths.audio_manifest,
            "graph_index_manifest": paths.graph_manifest,
            "candidate_policy_manifest": paths.candidate_policy,
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
