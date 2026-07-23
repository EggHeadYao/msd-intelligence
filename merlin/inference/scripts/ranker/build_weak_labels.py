"""Fit Set-A weak-label thresholds and build capped positive lists."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path

from ...artifact_paths import InferenceArtifactPaths
from ...catalog_data import load_catalog_context
from ...loaders import load_audio_index
from ...retrieval import TagRetriever
from ...scratch import prepare_scratch_root
from ...split import load_split_assignments, load_split_manifest
from ...tag_data import load_tag_idf
from ...training.weak_labels import (
    MAX_POSITIVES_PER_QUERY,
    fit_weak_label_thresholds,
    select_weak_positives,
    write_weak_positive_artifacts,
)


QUERY_BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    defaults = InferenceArtifactPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-assignments", type=Path, default=defaults.split_assignments)
    parser.add_argument("--split-manifest", type=Path, default=defaults.split_manifest)
    parser.add_argument("--thresholds", type=Path, default=defaults.weak_label_thresholds)
    parser.add_argument("--positives", type=Path, default=defaults.weak_positives)
    parser.add_argument("--manifest", type=Path, default=defaults.weak_positives_manifest)
    parser.add_argument("--query-splits", default="set_a")
    parser.add_argument("--max-threshold-pairs", type=int, default=1_000_000)
    parser.add_argument("--positive-neighbor-limit", type=int, default=1_001)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_threshold_pairs <= 0 or args.positive_neighbor_limit <= 0:
        raise ValueError("weak-label pair and neighbor limits must be positive")
    if args.limit_queries < 0:
        raise ValueError("limit-queries must be non-negative")
    paths = InferenceArtifactPaths()
    split_manifest = load_split_manifest(args.split_manifest, args.split_assignments)
    assignments = load_split_assignments(args.split_assignments)
    set_a = tuple(sorted(
        track_id for track_id, split in assignments.items() if split == "set_a"
    ))
    if not set_a:
        raise ValueError("split has no Set-A tracks for threshold fitting")
    query_splits = {value.strip() for value in args.query_splits.split(",") if value.strip()}
    if not query_splits.issubset({"set_a", "set_b", "set_c", "remaining"}):
        raise ValueError("query-splits contains an unknown split")

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
    thresholds = fit_weak_label_thresholds(
        set_a,
        tag.track_to_artist,
        audio.similarity,
        tag.pair_score,
        max_pairs=args.max_threshold_pairs,
        audio_batch_similarity=audio.pair_similarities,
    )
    thresholds["parent_split_scope"] = split_manifest["scope"]
    queries = tuple(sorted(
        track_id for track_id, split in assignments.items() if split in query_splits
    ))
    if args.limit_queries:
        queries = tuple(islice(queries, args.limit_queries))
    allowed_by_split: dict[str, set[str]] = {}
    for track_id, split in assignments.items():
        allowed_by_split.setdefault(split, set()).add(track_id)

    weak_audio_index = None
    weak_audio_tracks: tuple[str, ...] = ()
    if query_splits == {"set_a"}:
        import faiss

        weak_audio_tracks = tuple(sorted(allowed_by_split["set_a"]))
        weak_audio_index = faiss.IndexFlatIP(audio.dimension)
        weak_audio_index.add(audio.reconstruct_many(weak_audio_tracks))

    def audio_neighbor_batches(batch: tuple[str, ...]):
        if weak_audio_index is None:
            return audio.search_many(batch, args.positive_neighbor_limit)
        result_limit = min(args.positive_neighbor_limit, len(weak_audio_tracks))
        scores, row_ids = weak_audio_index.search(
            audio.reconstruct_many(batch), result_limit
        )
        return [
            sorted(
                (
                    (weak_audio_tracks[int(row_id)], float(score))
                    for row_id, score in zip(rows, values, strict=True)
                ),
                key=lambda item: (-item[1], item[0]),
            )
            for values, rows in zip(scores, row_ids, strict=True)
        ]

    def tag_neighbors(query_id: str) -> list[tuple[str, float]]:
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

    def records():
        for start in range(0, len(queries), QUERY_BATCH_SIZE):
            batch = queries[start : start + QUERY_BATCH_SIZE]
            audio_neighbors = audio_neighbor_batches(batch)
            for query_id, neighbors in zip(batch, audio_neighbors, strict=True):
                split = assignments[query_id]
                artist = tag.track_to_artist.get(query_id)
                positives = select_weak_positives(
                    query_id,
                    allowed_by_split[split],
                    tag.track_to_artist,
                    tag.artist_tracks.get(artist, ()) if artist else (),
                    neighbors,
                    tag_neighbors(query_id),
                    same_song,
                    thresholds,
                    limit=MAX_POSITIVES_PER_QUERY,
                )
                yield {
                    "query_track_id": query_id,
                    "split": split,
                    "positives": positives,
                }
            processed = min(start + len(batch), len(queries))
            if processed == len(queries) or processed % (10 * QUERY_BATCH_SIZE) == 0:
                print(
                    f"weak_labels_progress queries={processed}/{len(queries)}",
                    flush=True,
                )

    scope = "smoke" if args.limit_queries or split_manifest["scope"] == "smoke" else "formal"
    projected_gb = len(queries) * MAX_POSITIVES_PER_QUERY * 64 / (1024 ** 3)
    prepare_scratch_root(
        args.positives.parent,
        scope=scope,
        min_free_gb=args.min_free_gb,
        projected_gb=projected_gb,
    )
    manifest = write_weak_positive_artifacts(
        records(),
        args.positives,
        args.manifest,
        args.thresholds,
        thresholds,
        parent_paths={
            "split_manifest": args.split_manifest,
            "split_assignments": args.split_assignments,
            "audio_index_manifest": paths.audio_manifest,
            "tag_idf": paths.tag_idf,
            "songs_metadata": paths.songs_metadata,
        },
        scope=scope,
    )
    print(
        "weak_positives_ready "
        f"scope={scope} queries={manifest['query_count']} "
        f"positives={manifest['positive_count']} output={args.positives}",
    )


if __name__ == "__main__":
    main()
