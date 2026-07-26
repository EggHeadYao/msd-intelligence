        same_song=load_same_song_filter(paths.songs_metadata),
        ranker=ranker,
        fills=FeatureFillValues.from_artifact(paths.ranker_scaler, ranker.feature_order),
        candidate_policy=policy,
    )
