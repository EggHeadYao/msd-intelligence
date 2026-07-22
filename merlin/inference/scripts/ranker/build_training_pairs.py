"""CLI to construct split-safe candidate-aware Ranker pairs."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

from ...artifact_paths import InferenceArtifactPaths
from ...catalog_data import load_catalog_context
from ...candidate_pool import load_candidate_pool_manifest
from ...loaders import load_audio_index
from ...retrieval import TagRetriever
from ...scratch import prepare_scratch_root
from ...split import load_split_assignments
from ...tag_data import load_tag_idf
from ...training_pairs import write_training_pair_artifacts
from ...weak_labels import load_weak_positive_manifest, load_weak_positives


def parse_args() -> argparse.Namespace:
    defaults = InferenceArtifactPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, default=defaults.candidate_pool)
    parser.add_argument("--candidate-pool-manifest", type=Path, default=defaults.candidate_pool_manifest)
    parser.add_argument("--weak-positives", type=Path, default=defaults.weak_positives)
    parser.add_argument("--weak-positives-manifest", type=Path, default=defaults.weak_positives_manifest)
    parser.add_argument("--thresholds", type=Path, default=defaults.weak_label_thresholds)
    parser.add_argument("--split-assignments", type=Path, default=defaults.split_assignments)
    parser.add_argument("--output", type=Path, default=defaults.training_pairs)
    parser.add_argument("--manifest", type=Path, default=defaults.training_pairs_manifest)
    parser.add_argument("--stage", choices=("tuning", "final_retrain"), default="tuning")
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
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
    load_weak_positive_manifest(
        args.weak_positives_manifest,
        args.weak_positives,
        args.thresholds,
        expected_scope=args.scope,
    )
    assignments = load_split_assignments(args.split_assignments)
    positives = load_weak_positives(args.weak_positives)
    positive_count = sum(len(values) for values in positives.values())
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

    @lru_cache(maxsize=500_000)
    def is_positive(query_id: str, candidate_id: str) -> bool:
        query_artist = tag.track_to_artist.get(query_id)
        candidate_artist = tag.track_to_artist.get(candidate_id)
        if query_artist is not None and query_artist == candidate_artist:
            return True
        if query_artist is None or candidate_artist is None:
            return False
        audio_score = audio.similarity(query_id, candidate_id)
        if audio_score is not None and audio_score >= audio_threshold:
            return True
        tag_score = tag.pair_score(query_id, candidate_id)
        return tag_score is not None and tag_score >= tag_threshold

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
        positives,
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
