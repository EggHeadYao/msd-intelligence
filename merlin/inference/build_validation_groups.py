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
