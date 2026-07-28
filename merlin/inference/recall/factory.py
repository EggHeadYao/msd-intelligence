"""Production construction for canonical four-source candidate recall."""

from __future__ import annotations

import gc
from typing import Any, Mapping

from ..artifacts.paths import InferenceArtifactPaths
from ..data.graph import load_artist_neighbors
from .policy import (
    CANONICAL_BFS_MAX_DEPTH,
    CANONICAL_BFS_PER_ARTIST_CAP,
    CANONICAL_TAG_ARTIST_NEIGHBOR_LIMIT,
    CANONICAL_TAG_MAX_TERM_ARTISTS,
    CANONICAL_TAG_PER_ARTIST_CAP,
    CANONICAL_VECTOR_OVERFETCH_FACTOR,
    load_candidate_policy,
)
from ..data.catalog import SameSongFilter, load_catalog_context, load_same_song_filter
from .streaming import StreamingRecallEngine, TrackCodec
from ..retrieval.faiss import FaissTrackIndex, load_audio_index
from .pipeline import (
    RecallPipeline,
    allocate_backfill_groups,
    audit_recall_groups,
    candidate_digest,
    recall_query_report,
)
from ..retrieval import BfsRetriever, TagRetriever, VectorRetriever
from ..data.tags import load_tag_idf


def build_canonical_retrievers(
    audio: FaissTrackIndex,
    graph: FaissTrackIndex,
    paths: InferenceArtifactPaths,
    same_song: SameSongFilter,
    tag: TagRetriever | None = None,
) -> tuple[VectorRetriever, VectorRetriever, BfsRetriever, TagRetriever]:
    """Build all four retrievers with the frozen ordering and filtering policy."""
    if tag is None:
        tag = TagRetriever.from_parquet(
            str(paths.songs_metadata),
            str(paths.graph_edges),
            tag_idf_path=str(paths.tag_idf),
            same_song=same_song,
            artist_neighbor_limit=CANONICAL_TAG_ARTIST_NEIGHBOR_LIMIT,
            max_term_artists=CANONICAL_TAG_MAX_TERM_ARTISTS,
            per_artist_cap=CANONICAL_TAG_PER_ARTIST_CAP,
        )
    bfs = BfsRetriever(
        tag.track_to_artist,
        load_artist_neighbors(paths.graph_edges),
        tag.artist_tracks,
        same_song=same_song,
        tag_similarity=tag.artist_similarity,
        max_depth=CANONICAL_BFS_MAX_DEPTH,
        per_artist_cap=CANONICAL_BFS_PER_ARTIST_CAP,
    )
    return (
        VectorRetriever(
            "audio",
            audio.search,
            batch_search=audio.search_many,
            same_song=same_song,
            query_available=audio.contains,
            overfetch_factor=CANONICAL_VECTOR_OVERFETCH_FACTOR,
        ),
        VectorRetriever(
            "graph",
            graph.search,
            batch_search=graph.search_many,
            same_song=same_song,
            query_available=graph.contains,
            overfetch_factor=CANONICAL_VECTOR_OVERFETCH_FACTOR,
        ),
        bfs,
        tag,
    )


def load_recall_pipeline(
    paths: InferenceArtifactPaths = InferenceArtifactPaths(),
    *,
    graph_contract_key: str,
    graph_contract_version: str,
) -> RecallPipeline:
    """Load only Stage-1 artifacts; no Ranker bundle is required."""
    policy: dict[str, Any] = load_candidate_policy(paths.candidate_policy)
    audio = load_audio_index(
        paths.audio_index,
        paths.audio_mapping,
        paths.audio_manifest,
        paths.audio_encoder_metadata,
    )
    graph = FaissTrackIndex.from_files(
        paths.graph_index,
        paths.graph_mapping,
        paths.graph_manifest,
        paths.graph_encoder_metadata,
        expected_space="graph",
        expected_contract_key=graph_contract_key,
        expected_contract=graph_contract_version,
    )
    catalog = load_catalog_context(paths.songs_metadata, paths.graph_edges)
    same_song = catalog.same_song
    tag = TagRetriever.from_data(
        catalog.tag_data,
        idf_values=load_tag_idf(
            paths.tag_idf,
            expected_graph_edges_path=paths.graph_edges,
        ),
        same_song=same_song,
        artist_neighbor_limit=CANONICAL_TAG_ARTIST_NEIGHBOR_LIMIT,
        max_term_artists=CANONICAL_TAG_MAX_TERM_ARTISTS,
        per_artist_cap=CANONICAL_TAG_PER_ARTIST_CAP,
    )
    retrievers = build_canonical_retrievers(audio, graph, paths, same_song, tag)
    return RecallPipeline(
        retrievers=retrievers,
        retriever_limits={
            str(name): int(limit)
            for name, limit in policy["retriever_limits"].items()
        },
        candidate_limit=int(policy["candidate_limit"]),
        canonical=True,
        backfill_limits={
            str(name): int(limit)
            for name, limit in policy["backfill_limits"].items()
        },
        backfill_order=tuple(
            str(name) for name in policy["backfill_order"]
        ),
    )


def load_streaming_recall_engine(
    assignments: Mapping[str, str],
    allowed_splits: frozenset[str],
    paths: InferenceArtifactPaths = InferenceArtifactPaths(),
    *,
    graph_contract_key: str,
    graph_contract_version: str,
) -> StreamingRecallEngine:
    """Load the streaming batch engine used by high-volume recall stages."""
    policy: dict[str, Any] = load_candidate_policy(paths.candidate_policy)
    audio = load_audio_index(
        paths.audio_index,
        paths.audio_mapping,
        paths.audio_manifest,
        paths.audio_encoder_metadata,
    )
    graph = FaissTrackIndex.from_files(
        paths.graph_index,
        paths.graph_mapping,
        paths.graph_manifest,
        paths.graph_encoder_metadata,
        expected_space="graph",
        expected_contract_key=graph_contract_key,
        expected_contract=graph_contract_version,
    )
    catalog = load_catalog_context(paths.songs_metadata, paths.graph_edges)
    tag = TagRetriever.from_data(
        catalog.tag_data,
        idf_values=load_tag_idf(
            paths.tag_idf,
            expected_graph_edges_path=paths.graph_edges,
        ),
        same_song=catalog.same_song,
        artist_neighbor_limit=CANONICAL_TAG_ARTIST_NEIGHBOR_LIMIT,
        max_term_artists=CANONICAL_TAG_MAX_TERM_ARTISTS,
        per_artist_cap=CANONICAL_TAG_PER_ARTIST_CAP,
    )
    _audio, _graph, bfs, tag = build_canonical_retrievers(
        audio,
        graph,
        paths,
        catalog.same_song,
        tag,
    )
    return StreamingRecallEngine(
        audio,
        graph,
        bfs,
        tag,
        TrackCodec.build(
            assignments,
            allowed_splits,
            catalog.same_song,
            tag.track_to_artist,
        ),
        {
            str(name): int(limit)
            for name, limit in policy["retriever_limits"].items()
        },
        {
            str(name): int(limit)
            for name, limit in policy["backfill_limits"].items()
        },
        tuple(str(name) for name in policy["backfill_order"]),
        int(policy["candidate_limit"]),
    )


def validate_recall_low_memory(
    query_track_ids: tuple[str, ...],
    paths: InferenceArtifactPaths = InferenceArtifactPaths(),
    *,
    graph_contract_key: str,
    graph_contract_version: str,
) -> dict[str, object]:
    """Validate sources sequentially when two full FAISS indexes cannot coexist."""
    if not query_track_ids:
        raise ValueError("recall validation requires at least one query")
    policy: dict[str, Any] = load_candidate_policy(paths.candidate_policy)
    limits = {
        str(name): int(limit)
        for name, limit in policy["retriever_limits"].items()
    }
    backfill_limits = {
        str(name): int(limit)
        for name, limit in policy["backfill_limits"].items()
    }
    backfill_order = tuple(str(name) for name in policy["backfill_order"])
    groups: dict[str, dict[str, list[Any]]] = {
        query_id: {} for query_id in query_track_ids
    }
    availability: dict[str, dict[str, bool]] = {
        query_id: {} for query_id in query_track_ids
    }
    same_song = load_same_song_filter(paths.songs_metadata)

    def capture(retriever: Any) -> None:
        for query_id in query_track_ids:
            available = bool(retriever.is_available(query_id))
            first = (
                list(retriever.retrieve(query_id, backfill_limits[retriever.name]))
                if available
                else []
            )
            second = (
                list(retriever.retrieve(query_id, backfill_limits[retriever.name]))
                if available
                else []
            )
            if candidate_digest(first) != candidate_digest(second):
                raise ValueError(
                    f"{retriever.name} recall is not deterministic for {query_id}"
                )
            groups[query_id][retriever.name] = first
            availability[query_id][retriever.name] = available

    audio_index = load_audio_index(
        paths.audio_index,
        paths.audio_mapping,
        paths.audio_manifest,
        paths.audio_encoder_metadata,
    )
    capture(
        VectorRetriever(
            "audio",
            audio_index.search,
            batch_search=audio_index.search_many,
            same_song=same_song,
            query_available=audio_index.contains,
            overfetch_factor=CANONICAL_VECTOR_OVERFETCH_FACTOR,
        )
    )
    del audio_index
    gc.collect()

    graph_index = FaissTrackIndex.from_files(
        paths.graph_index,
        paths.graph_mapping,
        paths.graph_manifest,
        paths.graph_encoder_metadata,
        expected_space="graph",
        expected_contract_key=graph_contract_key,
        expected_contract=graph_contract_version,
    )
    capture(
        VectorRetriever(
            "graph",
            graph_index.search,
            batch_search=graph_index.search_many,
            same_song=same_song,
            query_available=graph_index.contains,
            overfetch_factor=CANONICAL_VECTOR_OVERFETCH_FACTOR,
        )
    )
    del graph_index
    gc.collect()

    tag = TagRetriever.from_parquet(
        str(paths.songs_metadata),
        str(paths.graph_edges),
        tag_idf_path=str(paths.tag_idf),
        same_song=same_song,
        artist_neighbor_limit=CANONICAL_TAG_ARTIST_NEIGHBOR_LIMIT,
        max_term_artists=CANONICAL_TAG_MAX_TERM_ARTISTS,
        per_artist_cap=CANONICAL_TAG_PER_ARTIST_CAP,
    )
    capture(tag)
    bfs = BfsRetriever(
        tag.track_to_artist,
        load_artist_neighbors(paths.graph_edges),
        tag.artist_tracks,
        same_song=same_song,
        tag_similarity=tag.artist_similarity,
        max_depth=CANONICAL_BFS_MAX_DEPTH,
        per_artist_cap=CANONICAL_BFS_PER_ARTIST_CAP,
    )
    capture(bfs)

    reports = []
    for query_id in query_track_ids:
        allocated_groups = allocate_backfill_groups(
            groups[query_id],
            limits,
            backfill_limits,
            backfill_order,
            int(policy["candidate_limit"]),
        )
        candidates, audit = audit_recall_groups(
            allocated_groups,
            limits,
            int(policy["candidate_limit"]),
            query_id,
            availability[query_id],
        )
        reports.append(recall_query_report(query_id, candidates, audit))
    return {
        "validation_status": "PASS",
        "validation_type": "structural_recall_audit",
        "execution_mode": "low_memory_sequential_sources",
        "candidate_recall_metrics_available": False,
        "query_count": len(reports),
        "queries": reports,
    }
