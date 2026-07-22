        paths.audio_manifest,
        paths.audio_encoder_metadata,
    )
    capture(
        VectorRetriever(
            "audio",
            audio_index.search,
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
        candidates, audit = audit_recall_groups(
            groups[query_id],
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
