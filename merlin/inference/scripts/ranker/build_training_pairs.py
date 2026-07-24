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
    parser.add_argument("--output", type=Path, default=defaults.training_pairs)
    parser.add_argument("--manifest", type=Path, default=defaults.training_pairs_manifest)
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


def main() -> None:
    args = parse_args()
    paths = InferenceArtifactPaths()
    load_candidate_pool_manifest(
        args.candidate_pool_manifest,
        args.candidate_pool,
        expected_scope=args.scope,
    )
    weak_positive_manifest = load_weak_positive_manifest(
        args.weak_positives_manifest,
        args.weak_positives,
        args.thresholds,
        expected_scope=args.scope,
    )
    assignments = load_split_assignments(args.split_assignments)
    positive_count = int(weak_positive_manifest.get("positive_count", 0))
    if positive_count <= 0:
        raise ValueError("weak-positive manifest has no positive rows")
    projected_gb = positive_count * 4 * 80 / (1024 ** 3)
    prepare_scratch_root(
        args.output.parent,
        scope=args.scope,
        min_free_gb=args.min_free_gb,
        projected_gb=projected_gb,
    )
    with args.thresholds.open("r", encoding="utf-8") as stream:
        thresholds = json.load(stream)
    audio_threshold = float(thresholds["audio_cosine_p90"])
    tag_threshold = float(thresholds["tag_tfidf_cosine_p90"])
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

    # Pair decisions are reused within one query while constructing candidate-
    # aware and random negatives.  Keeping this cache query-local prevents a
    # formal run from retaining millions of historical pair keys indefinitely.
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

    def is_positive_batch(query_id: str, candidate_ids: list[str]) -> list[bool]:
        query_artist = tag.track_to_artist.get(query_id)
        audio_scores = audio.similarities(query_id, candidate_ids)
        results = []
        for candidate_id, audio_score in zip(candidate_ids, audio_scores, strict=True):
            candidate_artist = tag.track_to_artist.get(candidate_id)
            if query_artist is not None and query_artist == candidate_artist:
                results.append(True)
                continue
            if query_artist is None or candidate_artist is None:
                results.append(False)
                continue
            if audio_score is not None and audio_score >= audio_threshold:
                results.append(True)
                continue
            tag_score = tag.pair_score(query_id, candidate_id)
            results.append(tag_score is not None and tag_score >= tag_threshold)
        return results

    manifest = write_training_pair_artifacts(
        args.candidate_pool,
        args.weak_positives,
        assignments,
        args.output,
        args.manifest,
        stage=args.stage,
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
        f"output={args.output}",
    )


if __name__ == "__main__":
    main()
