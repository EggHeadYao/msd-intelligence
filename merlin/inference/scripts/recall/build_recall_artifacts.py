"""Build the frozen candidate-policy and artist-term IDF artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...artifact_paths import CANDIDATE_POLICY_PATH, GRAPH_EDGES_PATH, TAG_IDF_PATH
from ...candidate_policy import write_candidate_policy
from ...tag_data import (
    build_tag_idf_artifact,
    load_artist_term_data,
    write_tag_idf_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-edges", type=Path, default=GRAPH_EDGES_PATH)
    parser.add_argument("--candidate-policy", type=Path, default=CANDIDATE_POLICY_PATH)
    parser.add_argument("--tag-idf", type=Path, default=TAG_IDF_PATH)
    parser.add_argument(
        "--parquet-engine",
        choices=("auto", "duckdb", "pyarrow"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_output_available(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"recall artifact already exists: {path}")


def main() -> None:
    args = parse_args()
    if not args.graph_edges.is_dir():
        raise FileNotFoundError(f"graph edge dataset does not exist: {args.graph_edges}")
    require_output_available(args.candidate_policy, args.overwrite)
    require_output_available(args.tag_idf, args.overwrite)

    tag_data = load_artist_term_data(
        args.graph_edges,
        parquet_engine=args.parquet_engine,
    )
    tag_idf = build_tag_idf_artifact(tag_data, args.graph_edges)
    write_tag_idf_artifact(tag_idf, args.tag_idf)
    write_candidate_policy(args.candidate_policy)
    print(
        "recall_artifacts_ready "
        f"artists={tag_idf['artist_count']} terms={tag_idf['term_count']} "
        f"tag_idf={args.tag_idf} candidate_policy={args.candidate_policy}",
    )


if __name__ == "__main__":
    main()
