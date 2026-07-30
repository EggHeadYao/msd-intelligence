"""CLI to build deterministic group-aware C3 split artifacts."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from ...artifacts.paths import (
    SONGS_METADATA_PATH,
    SPLIT_ASSIGNMENTS_PATH,
    SPLIT_MANIFEST_PATH,
)
from ...artifacts.io import parquet_rows
from ...training.split import build_split_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--songs-metadata", type=Path, default=SONGS_METADATA_PATH)
    parser.add_argument("--assignments", type=Path, default=SPLIT_ASSIGNMENTS_PATH)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    rows = parquet_rows(
        args.songs_metadata,
        ("track_id", "song_id"),
        order_by=("track_id",),
    )
    if args.limit:
        rows = islice(rows, args.limit)
    manifest = build_split_artifacts(
        rows,
        args.assignments,
        args.manifest,
        songs_metadata_path=args.songs_metadata,
        scope="smoke" if args.limit else "formal",
    )
    print(
        "split_ready "
        f"scope={manifest['scope']} tracks={manifest['track_count']} "
        f"groups={manifest['group_count']} assignments={args.assignments}",
    )


if __name__ == "__main__":
    main()
