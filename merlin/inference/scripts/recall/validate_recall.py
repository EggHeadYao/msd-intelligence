"""CLI for deterministic ranker-independent four-source recall validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from merlin.embedding.graph.config import GRAPH_CONTRACT_KEY, GRAPH_CONTRACT_VERSION

from ...artifact_paths import RECALL_VALIDATION_PATH
from ...recall import validate_recall_pipeline, write_recall_report
from ...recall_factory import load_recall_pipeline, validate_recall_low_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--graph-contract-key", default=GRAPH_CONTRACT_KEY)
    parser.add_argument("--graph-contract-version", default=GRAPH_CONTRACT_VERSION)
    parser.add_argument("--output", type=Path, default=RECALL_VALIDATION_PATH)
    parser.add_argument("--low-memory", action="store_true")
    return parser.parse_args()


def read_queries(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"query list does not exist: {path}")
    queries = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not queries:
        raise ValueError("query list must not be empty")
    if len(set(queries)) != len(queries):
        raise ValueError("query list contains duplicate track IDs")
    return queries


def main() -> None:
    args = parse_args()
    queries = read_queries(args.queries)
    if args.low_memory:
        report = validate_recall_low_memory(
            queries,
            graph_contract_key=args.graph_contract_key,
            graph_contract_version=args.graph_contract_version,
        )
    else:
        pipeline = load_recall_pipeline(
            graph_contract_key=args.graph_contract_key,
            graph_contract_version=args.graph_contract_version,
        )
        report = validate_recall_pipeline(pipeline, queries)
    write_recall_report(report, args.output)
    print(
        "recall_validation_passed "
        f"queries={report['query_count']} output={args.output}",
    )


if __name__ == "__main__":
    main()
