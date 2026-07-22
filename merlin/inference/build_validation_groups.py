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
