"""CLI to persist the canonical four-source candidate pool."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from merlin.embedding.graph.config import GRAPH_CONTRACT_KEY, GRAPH_CONTRACT_VERSION

from ...recall.policy import CANONICAL_CANDIDATE_LIMIT
from ...artifacts.paths import (
    CANDIDATE_POOL_MANIFEST_PATH,
    CANDIDATE_POOL_PATH,
    InferenceArtifactPaths,
)
from ...recall.pool import export_candidate_pool
from ...artifacts.integrity import sha256_path
from ...evaluation.protocol import load_development_protocol
from ...training.split import load_split_assignments, load_split_manifest
from ...recall.factory import load_streaming_recall_engine, load_recall_pipeline
from ..support.scratch import prepare_scratch_root
from .validate_recall import read_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queries", type=Path)
    source.add_argument("--split-assignments", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--query-split", choices=("set_a", "set_b", "set_c"))
    parser.add_argument("--development-protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--graph-contract-key", default=GRAPH_CONTRACT_KEY)
    parser.add_argument("--graph-contract-version", default=GRAPH_CONTRACT_VERSION)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_queries < 0:
        raise ValueError("limit-queries must be non-negative")
    paths = InferenceArtifactPaths()
    split_scope = None
    assignments = None
    parent_paths = {
        "audio_index_manifest": paths.audio_manifest,
        "graph_index_manifest": paths.graph_manifest,
        "candidate_policy_manifest": paths.candidate_policy,
        "tag_idf": paths.tag_idf,
        "songs_metadata": paths.songs_metadata,
        "graph_edges": paths.graph_edges,
    }
    if args.queries is not None:
        if (
            args.query_split is not None
            or args.split_manifest is not None
            or args.development_protocol is not None
        ):
            raise ValueError("query-split/split-manifest require split-assignments")
        queries = read_queries(args.queries)
        parent_paths["query_source"] = args.queries
    else:
        if args.query_split is None or args.split_manifest is None:
            raise ValueError("split-assignments require query-split and split-manifest")
        split = load_split_manifest(args.split_manifest, args.split_assignments)
        split_scope = str(split["scope"])
        assignments = load_split_assignments(args.split_assignments)
        queries = tuple(sorted(
            track_id for track_id, assignment in assignments.items()
            if assignment == args.query_split
        ))
        if not queries:
            raise ValueError(f"split contains no {args.query_split} queries")
        parent_paths["split_manifest"] = args.split_manifest
        parent_paths["split_assignments"] = args.split_assignments
    if args.limit_queries:
        queries = tuple(islice(queries, args.limit_queries))
    is_set_b = args.query_split == "set_b"
    is_development = args.query_split == "set_c"
    output = args.output or (
        paths.development_candidate_pool
        if is_development
        else paths.set_b_candidate_pool if is_set_b else CANDIDATE_POOL_PATH
    )
    manifest_path = args.manifest or (
        paths.development_candidate_pool_manifest
        if is_development
        else paths.set_b_candidate_pool_manifest
        if is_set_b
        else CANDIDATE_POOL_MANIFEST_PATH
    )
    scope = "smoke" if args.limit_queries or split_scope == "smoke" else "formal"
    if is_development:
        if args.development_protocol is None:
            raise ValueError("Set-C candidate export requires a development protocol")
        load_development_protocol(
            args.development_protocol,
            expected_scope=scope,
            expected_split="set_c",
            expected_parent_hashes={
                "split_manifest": sha256_path(args.split_manifest),
                "split_assignments": sha256_path(args.split_assignments),
                "candidate_policy_manifest": sha256_path(paths.candidate_policy),
                "validation_group_thresholds": sha256_path(
                    paths.validation_group_thresholds
                ),
                "ranker_training_manifest": sha256_path(
                    paths.ranker_training_manifest
                ),
                "no_hard_neg_training_manifest": sha256_path(
                    paths.no_hard_neg_training_manifest
                ),
                "audio_index_manifest": sha256_path(paths.audio_manifest),
                "graph_index_manifest": sha256_path(paths.graph_manifest),
                "tag_idf": sha256_path(paths.tag_idf),
                "songs_metadata": sha256_path(paths.songs_metadata),
                "graph_edges": sha256_path(paths.graph_edges),
            },
        )
        parent_paths["evaluation_protocol"] = args.development_protocol
    projected_gb = (
        len(queries) * CANONICAL_CANDIDATE_LIMIT * 64 / (1024 ** 3)
    )
    prepare_scratch_root(
        output.parent,
        scope=scope,
        min_free_gb=args.min_free_gb,
        projected_gb=projected_gb,
    )
    pipeline = (
        load_streaming_recall_engine(
            assignments,
            frozenset(assignments.values()),
            paths,
            graph_contract_key=args.graph_contract_key,
            graph_contract_version=args.graph_contract_version,
        )
        if assignments is not None
        else load_recall_pipeline(
            paths,
            graph_contract_key=args.graph_contract_key,
            graph_contract_version=args.graph_contract_version,
        )
    )
    manifest = export_candidate_pool(
        pipeline,
        queries,
        output,
        manifest_path,
        scope=scope,
        parent_paths=parent_paths,
    )
    print(
        "candidate_pool_ready "
        f"scope={manifest['scope']} queries={manifest['query_count']} "
        f"output={output}",
    )


if __name__ == "__main__":
    main()
