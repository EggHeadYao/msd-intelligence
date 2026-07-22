"""Evaluate C2 retrieval after masking each query's artist relation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from merlin.embedding.graph.build_faiss import INDEX_NAME, MAPPING_NAME
from merlin.embedding.graph.prepare_masked_artist_retrieval import EXPERIMENT_VERSION
from merlin.embedding.graph.retrieval_metrics import (
    DEFAULT_CUTOFFS,
    macro_average,
    random_expectation,
    score_ranking,
)


REPORT_VERSION = "c2_l1_2_report_v1"
REPORT_NAME = "report.json"
QUERY_METRICS_NAME = "query_metrics.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument("--overfetch", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def valid_song_key(track_id: str, song_id: str | None) -> str:
    if song_id is not None and song_id.strip():
        return f"song:{song_id}"
    return f"track:{track_id}"


def stable_rank_key(seed: int, query_id: str, candidate_id: str) -> bytes:
    value = f"{seed}\x00{query_id}\x00{candidate_id}".encode("ascii")
    return hashlib.sha256(value).digest()


def artist_size_slice(count: int) -> str:
    if count == 2:
        return "2"
    if count <= 5:
        return "3_5"
    if count <= 20:
        return "6_20"
    return "21_plus"


def release_degree_slice(count: int) -> str:
    if count <= 0:
        return "missing"
    if count == 1:
        return "singleton"
    if count <= 5:
        return "2_5"
    if count <= 20:
        return "6_20"
    return "21_plus"


def popularity_slice(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "missing"
    if value < 0.25:
        return "0_0.25"
    if value < 0.5:
        return "0.25_0.5"
    if value < 0.75:
        return "0.5_0.75"
    return "0.75_plus"


def load_queries(experiment: Path) -> list[dict[str, Any]]:
    table = pq.read_table(experiment / "queries.parquet")
    required = {
        "query_track_id",
        "artist_id",
        "song_id",
        "release_7digitalid",
        "song_hotttnesss",
        "artist_track_count",
        "positive_count",
        "release_degree",
        "candidate_catalog_size",
        "connectable",
        "stratum",
    }
    require(required <= set(table.column_names), "query manifest schema mismatch")
    require(table["query_track_id"].null_count == 0, "query track ID contains null")
    rows = table.to_pylist()
    query_ids = [str(row["query_track_id"]) for row in rows]
    require(len(query_ids) == len(set(query_ids)), "query track IDs are not unique")
    require(
        len({str(row["artist_id"]) for row in rows}) == len(rows),
        "query artists are not unique",
    )
    return rows


def load_positives(
    experiment: Path,
    queries: list[dict[str, Any]],
) -> dict[str, set[str]]:
    table = pq.read_table(experiment / "positives.parquet")
    require(
        table.column_names == ["query_track_id", "positive_track_id"],
        "positive pair schema mismatch",
    )
    positives: dict[str, set[str]] = defaultdict(set)
    for query_id, positive_id in zip(
        table["query_track_id"].to_pylist(),
        table["positive_track_id"].to_pylist(),
        strict=True,
    ):
        require(
            query_id is not None and positive_id is not None, "positive pair is null"
        )
        positives[str(query_id)].add(str(positive_id))

    query_ids = {str(row["query_track_id"]) for row in queries}
    require(set(positives) <= query_ids, "positives contain an unknown query")
    for row in queries:
        query_id = str(row["query_track_id"])
        require(
            len(positives[query_id]) == int(row["positive_count"]),
            f"positive count mismatch for {query_id}",
        )
    return positives


def load_mapping(graph_output: Path) -> tuple[list[str], dict[str, int]]:
    table = pq.read_table(graph_output / MAPPING_NAME)
    require(
        table.column_names == ["row_id", "node_id", "track_id"],
        "FAISS mapping schema mismatch",
    )
    row_ids = table["row_id"].combine_chunks().to_numpy(zero_copy_only=False)
    require(
        np.array_equal(row_ids, np.arange(table.num_rows)),
        "FAISS mapping row IDs are not contiguous",
    )
    row_to_track = [str(value) for value in table["track_id"].to_pylist()]
    require(len(row_to_track) == len(set(row_to_track)), "FAISS track mapping repeats")
    return row_to_track, {track_id: row for row, track_id in enumerate(row_to_track)}


def load_metadata(
    metadata_path: Path,
    query_releases: set[int],
) -> tuple[dict[str, str], dict[int, list[str]], int]:
    table = pq.read_table(
        metadata_path,
        columns=["track_id", "song_id", "release_7digitalid"],
    )
    track_to_song: dict[str, str] = {}
    release_to_tracks: dict[int, list[str]] = defaultdict(list)
    for batch in table.to_batches(max_chunksize=100_000):
        tracks = batch["track_id"].to_pylist()
        songs = batch["song_id"].to_pylist()
        releases = batch["release_7digitalid"].to_pylist()
        for track_value, song_value, release_value in zip(
            tracks,
            songs,
            releases,
            strict=True,
        ):
            require(track_value is not None, "metadata track ID contains null")
            track_id = str(track_value)
            require(track_id not in track_to_song, "metadata track ID is not unique")
            track_to_song[track_id] = valid_song_key(
                track_id,
                None if song_value is None else str(song_value),
            )
            if release_value is not None and int(release_value) in query_releases:
                release_to_tracks[int(release_value)].append(track_id)
    return track_to_song, release_to_tracks, table.num_rows


def search_c2(
    index: faiss.Index,
    queries: list[dict[str, Any]],
    track_to_row: dict[str, int],
    row_to_track: list[str],
    track_to_song: dict[str, str],
    max_cutoff: int,
    overfetch: int,
    batch_size: int,
) -> dict[str, list[str]]:
    connectable = [row for row in queries if bool(row["connectable"])]
    for row in connectable:
        query_id = str(row["query_track_id"])
        require(
            query_id in track_to_row,
            f"connectable query missing from FAISS: {query_id}",
        )
    rankings = {str(row["query_track_id"]): [] for row in queries}
    search_size = min(index.ntotal, max_cutoff * overfetch)

    for start in range(0, len(connectable), batch_size):
        batch = connectable[start : start + batch_size]
        vectors = np.stack(
            [
                index.reconstruct(track_to_row[str(row["query_track_id"])])
                for row in batch
            ],
        ).astype(np.float32, copy=False)
        scores, neighbors = index.search(vectors, search_size)
        for query, query_scores, query_neighbors in zip(
            batch,
            scores,
            neighbors,
            strict=True,
        ):
            query_id = str(query["query_track_id"])
            query_song = track_to_song[query_id]
            candidates: dict[str, float] = {}
            for score, row_id in zip(query_scores, query_neighbors, strict=True):
                if row_id < 0:
                    continue
                candidate_id = row_to_track[int(row_id)]
                if (
                    candidate_id == query_id
                    or track_to_song[candidate_id] == query_song
                ):
                    continue
                candidates.setdefault(candidate_id, float(score))
            rankings[query_id] = [
                track_id
                for track_id, _ in sorted(
                    candidates.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:max_cutoff]
            ]
    return rankings


def release_only_rankings(
    queries: list[dict[str, Any]],
    release_to_tracks: dict[int, list[str]],
    track_to_song: dict[str, str],
    max_cutoff: int,
    seed: int,
) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for query in queries:
        query_id = str(query["query_track_id"])
        if not bool(query["connectable"]):
            rankings[query_id] = []
            continue
        release = int(query["release_7digitalid"])
        query_song = track_to_song[query_id]
        candidates = {
            track_id
            for track_id in release_to_tracks.get(release, [])
            if track_id != query_id and track_to_song[track_id] != query_song
        }
        rankings[query_id] = sorted(
            candidates,
            key=lambda track_id: (stable_rank_key(seed, query_id, track_id), track_id),
        )[:max_cutoff]
    return rankings
