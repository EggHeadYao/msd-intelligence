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


def make_query_rows(
    queries: list[dict[str, Any]],
    positives: dict[str, set[str]],
    c2_rankings: dict[str, list[str]],
    release_rankings: dict[str, list[str]],
    random_candidate_counts: dict[str, int],
    cutoffs: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_track_id"])
        positive_set = positives[query_id]
        require(positive_set, f"query has no eligible positives: {query_id}")
        c2_metrics = score_ranking(c2_rankings[query_id], positive_set, cutoffs)
        release_metrics = score_ranking(
            release_rankings[query_id],
            positive_set,
            cutoffs,
        )
        random_metrics = random_expectation(
            random_candidate_counts[query_id],
            len(positive_set),
            cutoffs,
        )
        row: dict[str, Any] = {
            "query_track_id": query_id,
            "connectable": bool(query["connectable"]),
            "artist_size_slice": artist_size_slice(int(query["artist_track_count"])),
            "release_degree_slice": release_degree_slice(int(query["release_degree"])),
            "popularity_slice": popularity_slice(query["song_hotttnesss"]),
            "positive_count": len(positive_set),
            "random_candidate_count": random_candidate_counts[query_id],
            "c2_candidate_count": len(c2_rankings[query_id]),
            "release_candidate_count": len(release_rankings[query_id]),
        }
        for model, metrics in (
            ("c2", c2_metrics),
            ("release_only", release_metrics),
            ("random", random_metrics),
        ):
            for metric, value in metrics.items():
                row[f"{model}_{metric.replace('@', '_at_')}"] = value
        rows.append(row)
    return rows


def aggregate_rows(
    rows: list[dict[str, Any]],
    model: str,
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    metric_names = ("mrr",) + tuple(
        f"{metric}@{cutoff}"
        for cutoff in cutoffs
        for metric in ("recall", "hit", "ndcg")
    )
    metric_rows = [
        {
            name: float(row[f"{model}_{name.replace('@', '_at_')}"])
            for name in metric_names
        }
        for row in rows
    ]
    candidate_key = {
        "c2": "c2_candidate_count",
        "release_only": "release_candidate_count",
        "random": "random_candidate_count",
    }[model]
    result: dict[str, Any] = {
        "query_count": len(rows),
        "metrics": macro_average(metric_rows),
    }
    if rows and candidate_key in rows[0]:
        result["mean_candidate_count"] = sum(
            int(row[candidate_key]) for row in rows
        ) / len(rows)
        result["candidate_shortage"] = {
            str(cutoff): sum(int(row[candidate_key]) < cutoff for row in rows)
            for cutoff in cutoffs
        }
    return result


def model_report(
    rows: list[dict[str, Any]],
    model: str,
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    connectable = [row for row in rows if bool(row["connectable"])]
    report: dict[str, Any] = {
        "all_query": aggregate_rows(rows, model, cutoffs),
        "connectable_conditional": aggregate_rows(connectable, model, cutoffs),
        "slices": {},
    }
    for slice_name in (
        "artist_size_slice",
        "release_degree_slice",
        "popularity_slice",
    ):
        values = sorted({str(row[slice_name]) for row in rows})
        report["slices"][slice_name] = {
            value: aggregate_rows(
                [row for row in rows if row[slice_name] == value],
                model,
                cutoffs,
            )
            for value in values
        }
    return report


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"masked-artist evaluation output already exists: {output}"
        )
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    cutoffs = tuple(args.cutoffs)
    require(
        cutoffs == tuple(sorted(set(cutoffs))) and all(value > 0 for value in cutoffs),
        "cutoffs must be unique, positive, and increasing",
    )
    require(args.overfetch >= 1, "overfetch must be at least one")
    require(args.batch_size > 0, "batch size must be positive")

    config = read_json(args.experiment / "experiment_config.json")
    require(
        config.get("experiment_version") == EXPERIMENT_VERSION,
        "experiment version mismatch",
    )
    queries = load_queries(args.experiment)
    require(
        len(queries) == int(config["query_count"]),
        "query count does not match experiment config",
    )
    sorted_query_ids = sorted(str(row["query_track_id"]) for row in queries)
    query_hash = hashlib.sha256(
        ("\n".join(sorted_query_ids) + "\n").encode("ascii"),
    ).hexdigest()
    require(
        query_hash == config["query_track_ids_sha256"],
        "query manifest hash mismatch",
    )
    positives = load_positives(args.experiment, queries)

    index = faiss.read_index(str(args.graph_output / INDEX_NAME))
    require(index.metric_type == faiss.METRIC_INNER_PRODUCT, "FAISS metric mismatch")
    row_to_track, track_to_row = load_mapping(args.graph_output)
    require(index.ntotal == len(row_to_track), "FAISS and mapping row counts differ")

    query_releases = {
        int(row["release_7digitalid"])
        for row in queries
        if row["release_7digitalid"] is not None and int(row["release_7digitalid"]) > 0
    }
    metadata_path = args.experiment / "prepared" / "songs_metadata.parquet"
    track_to_song, release_to_tracks, metadata_rows = load_metadata(
        metadata_path,
        query_releases,
    )
    require(
        metadata_rows == int(config["counts"]["catalog_tracks"]),
        "metadata row count does not match experiment config",
    )
    require(
        all(track_id in track_to_song for track_id in row_to_track),
        "FAISS mapping contains a track absent from metadata",
    )

    max_cutoff = cutoffs[-1]
    c2_rankings = search_c2(
        index,
        queries,
        track_to_row,
        row_to_track,
        track_to_song,
        max_cutoff,
        args.overfetch,
        args.batch_size,
    )
    release_rankings = release_only_rankings(
        queries,
        release_to_tracks,
        track_to_song,
        max_cutoff,
        int(config["seed"]),
    )
    index_song_counts = Counter(track_to_song[track_id] for track_id in row_to_track)
    random_candidate_counts = {
        str(query["query_track_id"]): index.ntotal
        - index_song_counts[track_to_song[str(query["query_track_id"])]]
        for query in queries
    }
    rows = make_query_rows(
        queries,
        positives,
        c2_rankings,
        release_rankings,
        random_candidate_counts,
        cutoffs,
    )

    prepare_output(args.output, args.overwrite)
    pq.write_table(pa.Table.from_pylist(rows), args.output / QUERY_METRICS_NAME)
    connectable_count = sum(bool(row["connectable"]) for row in rows)
    report = {
        "report_version": REPORT_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_config": str(
            (args.experiment / "experiment_config.json").resolve()
        ),
        "graph_output": str(args.graph_output.resolve()),
        "parameters": {
            "cutoffs": list(cutoffs),
            "primary_cutoff": 20,
            "faiss_overfetch": args.overfetch,
            "mrr_scope": f"first positive within top {max_cutoff}",
            "release_only_tie_break": "sha256(seed, query_track_id, candidate_track_id)",
            "random_baseline": "exact uniform-without-replacement expectation",
        },
        "query_counts": {
            "total": len(rows),
            "without_positive": 0,
            "connectable": connectable_count,
            "unconnectable": len(rows) - connectable_count,
            "connectable_coverage": connectable_count / len(rows),
        },
        "models": {
            model: model_report(rows, model, cutoffs)
            for model in ("c2", "release_only", "random")
        },
        "claim_boundary": (
            "Transductive cross-relation metadata reconstruction; the masked query "
            "may recover artist context through release and is not strict "
            "relation-blind or inductive inference."
        ),
    }
