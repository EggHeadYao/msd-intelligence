    tag_threshold = float(thresholds["tag_tfidf_cosine_p90"])
    audio = load_audio_index()
    catalog = load_catalog_context(paths.songs_metadata, paths.graph_edges)
    same_song = catalog.same_song
    tag = TagRetriever.from_data(
        catalog.tag_data,
        idf_values=load_tag_idf(
            paths.tag_idf,
            expected_graph_edges_path=paths.graph_edges,
        ),
        same_song=same_song,
    )

    @lru_cache(maxsize=500_000)
    def is_positive(query_id: str, candidate_id: str) -> bool:
        query_artist = tag.track_to_artist.get(query_id)
        candidate_artist = tag.track_to_artist.get(candidate_id)
        if query_artist is not None and query_artist == candidate_artist:
            return True
        if query_artist is None or candidate_artist is None:
            return False
        audio_score = audio.similarity(query_id, candidate_id)
        if audio_score is not None and audio_score >= audio_threshold:
            return True
        tag_score = tag.pair_score(query_id, candidate_id)
        return tag_score is not None and tag_score >= tag_threshold

    def is_positive_batch(query_id: str, candidate_ids: list[str]) -> list[bool]:
        query_artist = tag.track_to_artist.get(query_id)
        audio_scores = audio.similarities(query_id, candidate_ids)
        results = []
        for candidate_id, audio_score in zip(candidate_ids, audio_scores, strict=True):
            candidate_artist = tag.track_to_artist.get(candidate_id)
            if query_artist is not None and query_artist == candidate_artist:
                results.append(True)
                continue
            if query_artist is None or candidate_artist is None:
                results.append(False)
                continue
            if audio_score is not None and audio_score >= audio_threshold:
                results.append(True)
                continue
            tag_score = tag.pair_score(query_id, candidate_id)
            results.append(tag_score is not None and tag_score >= tag_threshold)
        return results

    manifest = write_training_pair_artifacts(
        args.candidate_pool,
        positives,
        assignments,
        args.output,
        args.manifest,
        stage=args.stage,
        same_song=same_song,
        is_positive=is_positive,
        is_positive_batch=is_positive_batch,
        parent_paths={
            "candidate_pool": args.candidate_pool,
            "weak_positives": args.weak_positives,
            "weak_label_thresholds": args.thresholds,
            "split_assignments": args.split_assignments,
            "audio_index_manifest": paths.audio_manifest,
            "tag_idf": paths.tag_idf,
        },
        scope=args.scope,
    )
    print(
        "training_pairs_ready "
        f"scope={args.scope} stage={args.stage} pairs={manifest['pair_count']} "
        f"output={args.output}",
    )


if __name__ == "__main__":
    main()
