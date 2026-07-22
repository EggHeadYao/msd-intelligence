    else:
        if args.stage != "tuning":
            raise ValueError("validation features are only defined for tuning")
        load_validation_group_manifest(
            pairs_manifest,
            thresholds_path=args.validation_thresholds,
            positives_path=args.validation_positives,
            validation_pairs_path=pairs,
            expected_scope=args.scope,
        )
    audio = load_audio_index()
    graph = FaissTrackIndex.from_files(
        paths.graph_index,
        paths.graph_mapping,
        paths.graph_manifest,
        paths.graph_encoder_metadata,
        expected_space="graph",
        expected_contract_key=args.graph_contract_key,
        expected_contract=args.graph_contract_version,
    )
    catalog = load_catalog_context(
        paths.songs_metadata,
        paths.graph_edges,
        include_ranker_metadata=True,
    )
    same_song = catalog.same_song
    tag = TagRetriever.from_data(
        catalog.tag_data,
        idf_values=load_tag_idf(
            paths.tag_idf,
            expected_graph_edges_path=paths.graph_edges,
        ),
        same_song=same_song,
    )
    _audio, _graph, bfs, tag = build_canonical_retrievers(
        audio, graph, paths, same_song, tag
    )
    computer = RankerV2FeatureComputer(
        tracks=catalog.ranker_tracks,
        signals=PairSignalLookups(
            audio=audio.similarity,
            graph=graph.similarity,
            bfs=bfs.pair_score,
            tags=tag.pair_score,
            audio_batch=audio.similarities,
            graph_batch=graph.similarities,
        ),
    )
    manifest = export_raw_pair_features(
        pairs,
        computer,
        output,
        output_manifest,
        parent_paths={
            f"{args.pair_kind}_pairs": pairs,
            f"{args.pair_kind}_pairs_manifest": pairs_manifest,
            "audio_index_manifest": paths.audio_manifest,
            "graph_index_manifest": paths.graph_manifest,
            "tag_idf": paths.tag_idf,
            "songs_metadata": paths.songs_metadata,
            "graph_edges": paths.graph_edges,
        },
        scope=args.scope,
        pair_kind=args.pair_kind,
        stage=args.stage,
    )
    print(
        "ranker_features_ready "
        f"scope={args.scope} kind={args.pair_kind} "
        f"rows={manifest['row_count']} output={output}",
    )


if __name__ == "__main__":
    main()
