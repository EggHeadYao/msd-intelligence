"""CLI to persist the canonical four-source candidate pool."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from merlin.embedding.graph.config import GRAPH_CONTRACT_KEY, GRAPH_CONTRACT_VERSION

from ...artifact_paths import (
    CANDIDATE_POOL_MANIFEST_PATH,
    CANDIDATE_POOL_PATH,
    InferenceArtifactPaths,
)
from ...candidate_pool import export_candidate_pool
from ...split import load_split_assignments, load_split_manifest
from ...recall_factory import load_recall_pipeline
from .validate_recall import read_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queries", type=Path)
    source.add_argument("--split-assignments", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--query-split", choices=("set_a", "set_b"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--graph-contract-key", default=GRAPH_CONTRACT_KEY)
    parser.add_argument("--graph-contract-version", default=GRAPH_CONTRACT_VERSION)
    parser.add_argument("--limit-queries", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_queries < 0:
        raise ValueError("limit-queries must be non-negative")
    paths = InferenceArtifactPaths()
    split_scope = None
    parent_paths = {
        "audio_index_manifest": paths.audio_manifest,
        "graph_index_manifest": paths.graph_manifest,
        "candidate_policy_manifest": paths.candidate_policy,
        "tag_idf": paths.tag_idf,
        "songs_metadata": paths.songs_metadata,
        "graph_edges": paths.graph_edges,
    }
    if args.queries is not None:
        if args.query_split is not None or args.split_manifest is not None:
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
    output = args.output or (
        paths.set_b_candidate_pool if is_set_b else CANDIDATE_POOL_PATH
    )
    manifest_path = args.manifest or (
        paths.set_b_candidate_pool_manifest if is_set_b else CANDIDATE_POOL_MANIFEST_PATH
    )
    scope = "smoke" if args.limit_queries or split_scope == "smoke" else "formal"
    pipeline = load_recall_pipeline(
        paths,
        graph_contract_key=args.graph_contract_key,
        graph_contract_version=args.graph_contract_version,
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
