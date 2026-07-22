        tag.pair_score,
        max_pairs=args.max_threshold_pairs,
        audio_batch_similarity=audio.pair_similarities,
    )
    thresholds["parent_split_scope"] = split_manifest["scope"]
    queries = tuple(sorted(
        track_id for track_id, split in assignments.items() if split in query_splits
    ))
    if args.limit_queries:
        queries = tuple(islice(queries, args.limit_queries))
    allowed_by_split: dict[str, set[str]] = {}
    for track_id, split in assignments.items():
        allowed_by_split.setdefault(split, set()).add(track_id)

    def tag_neighbors(query_id: str) -> list[tuple[str, float]]:
        artist = tag.track_to_artist.get(query_id)
        if artist is None:
            return []
        artists = (
            tag.similar_artists(artist)
            if callable(tag.similar_artists)
            else tag.similar_artists.get(artist, ())
        )
        return [
            (track_id, float(score))
            for target_artist, score in artists
            for track_id in sorted(tag.artist_tracks.get(target_artist, ()))
        ]

    def records():
        for query_id in queries:
            split = assignments[query_id]
            artist = tag.track_to_artist.get(query_id)
            positives = select_weak_positives(
                query_id,
                allowed_by_split[split],
                tag.track_to_artist,
                tag.artist_tracks.get(artist, ()) if artist else (),
                audio.search(query_id, args.positive_neighbor_limit),
                tag_neighbors(query_id),
                same_song,
                thresholds,
                limit=MAX_POSITIVES_PER_QUERY,
            )
            yield {
                "query_track_id": query_id,
                "split": split,
                "positives": positives,
            }

    scope = "smoke" if args.limit_queries or split_manifest["scope"] == "smoke" else "formal"
    manifest = write_weak_positive_artifacts(
        records(),
        args.positives,
        args.manifest,
        args.thresholds,
        thresholds,
        parent_paths={
            "split_manifest": args.split_manifest,
            "split_assignments": args.split_assignments,
            "audio_index_manifest": paths.audio_manifest,
            "tag_idf": paths.tag_idf,
            "songs_metadata": paths.songs_metadata,
        },
        scope=scope,
    )
    print(
        "weak_positives_ready "
        f"scope={scope} queries={manifest['query_count']} "
        f"positives={manifest['positive_count']} output={args.positives}",
    )


if __name__ == "__main__":
    main()
