"""CLI for deterministic end-to-end C3 inference validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..artifacts.paths import INFERENCE_VALIDATION_PATH
from ..runtime.factory import load_inference_pipeline
from ..runtime.validation import validate_pipeline, write_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the final MERLIN C3 pipeline.")
    parser.add_argument("--queries", type=Path, required=True, help="One track_id per line.")
    parser.add_argument("--graph-contract-key", required=True)
    parser.add_argument("--graph-contract-version", required=True)
    parser.add_argument("--output", type=Path, default=INFERENCE_VALIDATION_PATH)
    parser.add_argument("--score-tolerance", type=float, default=1e-7)
    return parser.parse_args()


def read_queries(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"query list does not exist: {path}")
    queries = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not queries:
        raise ValueError("query list must not be empty")
    if len(set(queries)) != len(queries):
        raise ValueError("query list contains duplicate track IDs")
    return queries


def main() -> None:
    args = parse_args()
    pipeline = load_inference_pipeline(
        graph_contract_key=args.graph_contract_key,
        graph_contract_version=args.graph_contract_version,
    )
    report = validate_pipeline(
        pipeline,
        read_queries(args.queries),
        score_tolerance=args.score_tolerance,
    )
    write_validation_report(report, args.output)
    print(f"inference_validation_passed queries={report['query_count']} output={args.output}")


if __name__ == "__main__":
    main()
