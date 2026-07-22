        cached.append(threshold_sample)
        threshold_sample_count = threshold_sample.count()
        if threshold_sample_count == 0:
            raise ValueError("Set-A pre-PCA threshold sample is empty")
        percentiles = threshold_sample.select(
            F.percentile_approx("pre_pca_cosine", [0.5, 0.9], 1_000_000).alias("values")
        ).first()["values"]
        acoustic_p50, acoustic_p90 = (float(value) for value in percentiles)
        if not math.isfinite(acoustic_p50) or not math.isfinite(acoustic_p90):
            raise ValueError("Set-A pre-PCA thresholds are not finite")
        if acoustic_p90 < acoustic_p50:
            raise ValueError("Set-A pre-PCA thresholds are not monotonic")

        set_b = vectors.where(F.col("split") == "set_b").drop("split").persist(
            StorageLevel.MEMORY_AND_DISK
        )
        cached.append(set_b)
        query_b = set_b.join(F.broadcast(pool_queries), TRACK_ID_COLUMN, "inner")
        if query_b.limit(1).count() == 0:
            raise ValueError("candidate pool has no Set-B query with a C1 vector")

        b_artists = set_b.where(
            F.col("artist_id").isNotNull() & (F.length("artist_id") > 0)
        ).select("artist_id").distinct()
        idf_rows = [(term, float(value) ** 2) for term, value in sorted(tag_idf.items())]
        idf = spark.createDataFrame(idf_rows, ("term", "idf_squared"))
        terms = (
            spark.read.parquet(_uri(args.graph_edges / "edge_type=artist_term"))
            .select(F.col("src_id").cast("string").alias("artist_id"), F.col("dst_id").cast("string").alias("term"))
            .distinct()
            .join(F.broadcast(b_artists), "artist_id", "inner")
            .join(F.broadcast(idf), "term", "inner")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        cached.append(terms)
        norms = terms.groupBy("artist_id").agg(
            F.sqrt(F.sum("idf_squared")).alias("tag_norm")
        ).persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(norms)
        left_terms = terms.select(
            F.col("artist_id").alias("q_artist_id"), "term", "idf_squared"
        )
        right_terms = terms.select(
            F.col("artist_id").alias("c_artist_id"), "term"
        )
        tag_scores = (
            left_terms.join(right_terms, "term", "inner")
            .where(F.col("q_artist_id") != F.col("c_artist_id"))
            .groupBy("q_artist_id", "c_artist_id")
            .agg(F.sum("idf_squared").alias("tag_numerator"))
            .join(norms.select(F.col("artist_id").alias("q_artist_id"), F.col("tag_norm").alias("q_norm")), "q_artist_id")
            .join(norms.select(F.col("artist_id").alias("c_artist_id"), F.col("tag_norm").alias("c_norm")), "c_artist_id")
            .withColumn("tag_tfidf_cosine", F.col("tag_numerator") / (F.col("q_norm") * F.col("c_norm")))
            .select("q_artist_id", "c_artist_id", "tag_tfidf_cosine")
            .localCheckpoint(eager=True)
        )
        cached.append(tag_scores)
        tagged_artists = norms.select("artist_id")

        def q_columns(frame):
            return frame.select(
                F.col(TRACK_ID_COLUMN).alias("query_track_id"),
                F.col("song_id").alias("q_song_id"),
                F.col("artist_id").alias("q_artist_id"),
                F.col("release_id").alias("q_release_id"),
                F.col("pre_pca_vector").alias("q_vector"),
                F.col("pre_pca_norm").alias("q_norm"),
            )

        def c_columns(frame):
            return frame.select(
                F.col(TRACK_ID_COLUMN).alias("candidate_track_id"),
                F.col("song_id").alias("c_song_id"),
                F.col("artist_id").alias("c_artist_id"),
                F.col("release_id").alias("c_release_id"),
                F.col("pre_pca_vector").alias("c_vector"),
                F.col("pre_pca_norm").alias("c_norm"),
            )

        q_tracks = q_columns(query_b)
        c_tracks = c_columns(set_b)
        valid_q_audio = (
            q_tracks.where(
                F.col("q_artist_id").isNotNull()
                & (F.length("q_artist_id") > 0)
                & F.col("q_release_id").isNotNull()
                & (F.col("q_release_id") > 0)
                & (F.col("q_norm") > 0.0)
            )
            .join(
                tagged_artists.select(F.col("artist_id").alias("q_artist_id")),
                "q_artist_id",
                "inner",
            )
        )
        valid_c_audio = (
            c_tracks.where(
                F.col("c_artist_id").isNotNull()
                & (F.length("c_artist_id") > 0)
                & F.col("c_release_id").isNotNull()
                & (F.col("c_release_id") > 0)
                & (F.col("c_norm") > 0.0)
            )
            .join(
                tagged_artists.select(F.col("artist_id").alias("c_artist_id")),
                "c_artist_id",
                "inner",
            )
        )
        if args.audio_pair_engine == "numpy":
            audio_pair_temporary = TemporaryDirectory(prefix="merlin-setb-audio-pairs-")
            raw_audio_pairs_path = Path(audio_pair_temporary.name) / "pairs.parquet"
            write_audio_threshold_pairs_numpy(
                [row.asDict(recursive=True) for row in valid_q_audio.collect()],
                [row.asDict(recursive=True) for row in valid_c_audio.collect()],
                raw_audio_pairs_path,
                threshold=acoustic_p90,
                block_size=args.audio_block_size,
            )
            raw_audio_pairs = spark.read.parquet(_uri(raw_audio_pairs_path))
        else:
            raw_audio_pairs = (
                valid_q_audio.crossJoin(valid_c_audio)
                .where(F.col("query_track_id") != F.col("candidate_track_id"))
                .where(not_same_song("q", "c"))
                .where(
                    (F.col("q_artist_id") != F.col("c_artist_id"))
                    & (F.col("q_release_id") != F.col("c_release_id"))
                )
                .withColumn(
                    "pre_pca_cosine",
                    cosine(
                        F.col("q_vector"),
                        F.col("c_vector"),
                        F.col("q_norm"),
                        F.col("c_norm"),
                    ),
                )
                .where(F.col("pre_pca_cosine") >= F.lit(acoustic_p90))
                .select("query_track_id", "candidate_track_id", "q_artist_id", "c_artist_id")
            )
        audio_pairs = (
            raw_audio_pairs.join(tag_scores, ["q_artist_id", "c_artist_id"], "left")
            .fillna({"tag_tfidf_cosine": 0.0})
            .where(F.col("tag_tfidf_cosine") < F.lit(tag_positive_threshold))
            .select(
                "query_track_id",
                "candidate_track_id",
                F.lit("audio_dominant").alias("query_group"),
                F.array(F.lit("pre_pca_audio")).alias("positive_sources"),
            ).persist(StorageLevel.MEMORY_AND_DISK)
        )
        cached.append(audio_pairs)

        q_meta = q_tracks.drop("q_vector")
        c_meta = c_tracks.drop("c_vector")
        same_artist = q_meta.join(c_meta, F.col("q_artist_id") == F.col("c_artist_id"), "inner").select(
            "query_track_id", "candidate_track_id", F.lit("same_artist").alias("relation_source")
        )
        same_release = q_meta.where(F.col("q_release_id").isNotNull() & (F.col("q_release_id") > 0)).join(
            c_meta.where(F.col("c_release_id").isNotNull() & (F.col("c_release_id") > 0)),
            F.col("q_release_id") == F.col("c_release_id"),
            "inner",
        ).select("query_track_id", "candidate_track_id", F.lit("same_release").alias("relation_source"))
        directed_edges = spark.read.parquet(
            _uri(args.graph_edges / "edge_type=artist_similarity")
        ).select(
            F.col("src_id").cast("string").alias("q_artist_id"),
            F.col("dst_id").cast("string").alias("c_artist_id"),
        ).distinct()
        directed = q_meta.join(directed_edges, "q_artist_id", "inner").join(
            c_meta, "c_artist_id", "inner"
        ).select("query_track_id", "candidate_track_id", F.lit("directed_artist_similarity").alias("relation_source"))
        high_tag = tag_scores.where(
            F.col("tag_tfidf_cosine") >= F.lit(tag_positive_threshold)
        ).join(q_meta, "q_artist_id", "inner").join(c_meta, "c_artist_id", "inner").select(
            "query_track_id", "candidate_track_id", F.lit("high_artist_term").alias("relation_source")
        )
        relation_candidates = (
            same_artist.unionByName(same_release).unionByName(directed).unionByName(high_tag)
            .where(F.col("query_track_id") != F.col("candidate_track_id"))
            .groupBy("query_track_id", "candidate_track_id")
            .agg(F.sort_array(F.collect_set("relation_source")).alias("positive_sources"))
            .localCheckpoint(eager=True)
        )
        cached.append(relation_candidates)
        relation_pairs = (
            relation_candidates.join(q_tracks, "query_track_id", "inner")
            .join(c_tracks, "candidate_track_id", "inner")
            .where(not_same_song("q", "c"))
            .where((F.col("q_norm") > 0.0) & (F.col("c_norm") > 0.0))
            .withColumn(
                "pre_pca_cosine",
                cosine(F.col("q_vector"), F.col("c_vector"), F.col("q_norm"), F.col("c_norm")),
            )
            .where(F.col("pre_pca_cosine") < F.lit(acoustic_p50))
            .select(
                "query_track_id",
                "candidate_track_id",
                F.lit("relation_dominant").alias("query_group"),
                "positive_sources",
            ).persist(StorageLevel.MEMORY_AND_DISK)
        )
        cached.append(relation_pairs)
        audio_pairs.count()
        relation_pairs.count()

        counts = audio_pairs.groupBy("query_track_id").count().withColumnRenamed("count", "audio_count").join(
            relation_pairs.groupBy("query_track_id").count().withColumnRenamed("count", "relation_count"),
            "query_track_id",
            "inner",
        ).withColumn("balanced_count", F.least("audio_count", "relation_count")).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        cached.append(counts)
        audio_window = Window.partitionBy("query_track_id").orderBy(
            F.xxhash64("query_track_id", "candidate_track_id", F.lit("audio"), F.lit(VALIDATION_GROUP_SEED)),
            "candidate_track_id",
        )
        relation_window = Window.partitionBy("query_track_id").orderBy(
            F.xxhash64("query_track_id", "candidate_track_id", F.lit("relation"), F.lit(VALIDATION_GROUP_SEED)),
            "candidate_track_id",
        )
        mixed_audio = audio_pairs.withColumn("side_rank", F.row_number().over(audio_window)).join(
            counts.select("query_track_id", "balanced_count"), "query_track_id"
        ).where(F.col("side_rank") <= F.col("balanced_count")).select(
            "query_track_id",
            "candidate_track_id",
            F.lit("mixed").alias("query_group"),
            F.array(F.lit("audio_dominant_side")).alias("positive_sources"),
        )
        mixed_relation = relation_pairs.withColumn("side_rank", F.row_number().over(relation_window)).join(
            counts.select("query_track_id", "balanced_count"), "query_track_id"
        ).where(F.col("side_rank") <= F.col("balanced_count")).select(
            "query_track_id",
            "candidate_track_id",
            F.lit("mixed").alias("query_group"),
            F.array(F.lit("relation_dominant_side")).alias("positive_sources"),
        )
        positives = (
            audio_pairs.unionByName(relation_pairs).unionByName(mixed_audio).unionByName(mixed_relation)
            .dropDuplicates(["query_track_id", "candidate_track_id", "query_group"])
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        cached.append(positives)
        eligible = positives.groupBy("query_track_id", "query_group").agg(
            F.count("*").cast("long").alias("eligible_positive_count")
        ).persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(eligible)
        candidate_rows = pool.select(
            "query_track_id", F.explode("candidates").alias("candidate")
        ).select(
            "query_track_id",
            F.col("candidate.track_id").cast("string").alias("candidate_track_id"),
            F.col("candidate.recall_sources").alias("recall_sources"),
        )
        positive_keys = positives.select(
            "query_track_id", "candidate_track_id", "query_group"
        ).withColumn("label", F.lit(1))
        validation_pairs = (
            eligible.join(candidate_rows, "query_track_id", "inner")
            .join(positive_keys, ["query_track_id", "candidate_track_id", "query_group"], "left")
            .fillna({"label": 0})
            .select(
                "query_track_id",
                "candidate_track_id",
                F.col("label").cast("int"),
                "query_group",
                "eligible_positive_count",
                "recall_sources",
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        cached.append(validation_pairs)
        missing_candidate_queries = eligible.select("query_track_id", "query_group").join(
            validation_pairs.select("query_track_id", "query_group").distinct(),
            ["query_track_id", "query_group"],
            "left_anti",
        ).limit(1).count()
        if missing_candidate_queries:
            raise ValueError("an eligible Set-B validation query has no canonical candidates")

        positive_stats = {
            row["query_group"]: row.asDict()
            for row in positives.groupBy("query_group").agg(
                F.count("*").alias("positive_count"),
                F.countDistinct("query_track_id").alias("eligible_query_count"),
            ).collect()
        }
        hit_stats = {
            row["query_group"]: row.asDict()
            for row in validation_pairs.where(F.col("label") == 1).groupBy("query_group").agg(
                F.count("*").alias("candidate_hits"),
                F.countDistinct("query_track_id").alias("covered_queries"),
            ).collect()
        }
        group_stats = {}
        for group in VALIDATION_QUERY_GROUPS:
            positive_count = int(positive_stats.get(group, {}).get("positive_count", 0))
            eligible_query_count = int(
                positive_stats.get(group, {}).get("eligible_query_count", 0)
            )
            candidate_hits = int(hit_stats.get(group, {}).get("candidate_hits", 0))
            covered_queries = int(hit_stats.get(group, {}).get("covered_queries", 0))
            group_stats[group] = {
                "eligible_query_count": eligible_query_count,
                "positive_count": positive_count,
                "candidate_positive_hits": candidate_hits,
                "candidate_recall": candidate_hits / positive_count if positive_count else 0.0,
                "zero_coverage_query_count": eligible_query_count - covered_queries,
            }
        if args.scope == "formal" and any(
            int(group_stats[group]["eligible_query_count"]) == 0 for group in VALIDATION_QUERY_GROUPS
        ):
            raise ValueError("formal Set-B validation is missing an eligible query group")

        threshold_payload = {
            "artifact_type": "set_b_validation_thresholds",
            "artifact_version": "merlin_validation_groups_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "fit_split": "set_a",
            "seed": VALIDATION_GROUP_SEED,
            "sample_method": "deterministic_hash_sampled_cross_artist_pairs",
            "quantile_method": "percentile_approx_accuracy_1000000",
            "max_sample_pairs": args.max_threshold_pairs,
            "sampled_cross_artist_pairs": threshold_sample_count,
            "pre_pca_acoustic_cosine_p50": acoustic_p50,
            "pre_pca_acoustic_cosine_p90": acoustic_p90,
            "tag_positive_threshold": tag_positive_threshold,
            "tag_positive_threshold_source": str(args.weak_thresholds),
            "audio_pair_engine": args.audio_pair_engine,
            "audio_block_size": args.audio_block_size,
        }
        write_json_atomic(threshold_payload, args.thresholds)
        positives.write.mode("errorifexists").parquet(_uri(args.positives))
        validation_pairs.write.mode("errorifexists").parquet(_uri(args.validation_pairs))
        manifest = write_validation_group_manifest(
            args.manifest,
            thresholds_path=args.thresholds,
            positives_path=args.positives,
            validation_pairs_path=args.validation_pairs,
            parent_paths={
                "prepared_manifest": args.prepared_manifest,
                "c1_manifest": c1_manifest_path,
                "audio_encoder_metadata": encoder_metadata_path,
                "audio_scaler_model": scaler_model_path,
                "split_manifest": args.split_manifest,
                "split_assignments": args.split_assignments,
                "candidate_pool_manifest": args.candidate_pool_manifest,
                "candidate_pool": args.candidate_pool,
                "weak_label_thresholds": args.weak_thresholds,
                "tag_idf": args.tag_idf,
            },
            scope=args.scope,
            threshold_sample_count=threshold_sample_count,
            group_stats=group_stats,
        )
        print(
            "validation_groups_ready "
            f"scope={manifest['scope']} set_a_sample={threshold_sample_count} "
            f"candidate_queries={candidate_manifest['query_count']} output={args.validation_pairs}"
        )
    finally:
        for frame in reversed(cached):
            try:
                frame.unpersist(blocking=False)
            except Exception:
                pass
        spark.stop()
        if audio_pair_temporary is not None:
            audio_pair_temporary.cleanup()


if __name__ == "__main__":
    main()
