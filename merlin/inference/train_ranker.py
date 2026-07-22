                    "idcg20",
                    F.aggregate(
                        F.sequence(
                            F.lit(1),
                            F.least(F.col("positive_count").cast("int"), F.lit(20)),
                        ),
                        F.lit(0.0),
                        lambda total, rank: total
                        + 1.0 / F.log2(rank.cast("double") + 1.0),
                    ),
                )
                .withColumn("ndcg20", F.col("dcg") / F.col("idcg20"))
            )
            collected_by_reg = {reg_param: [] for reg_param in REG_PARAMS}
            for row in per_query.collect():
                collected_by_reg[float(row["reg_param"])].append(row)
            query_scores = {}
            for reg_param in REG_PARAMS:
                collected = collected_by_reg[reg_param]
                group_counts = {
                    group: sum(row["query_group"] == group for row in collected)
                    for group in QUERY_GROUPS
                }
                if any(count == 0 for count in group_counts.values()):
                    raise ValueError("Set-B validation is missing a frozen query group")
                total = len(collected)
                query_scores[reg_param] = [
                    float(row["ndcg20"])
                    * total
                    / (len(QUERY_GROUPS) * group_counts[row["query_group"]])
                    for row in sorted(
                        collected,
                        key=lambda item: (item["query_group"], item["query_track_id"]),
                    )
                ]
            selected_reg, selection = select_reg_param(query_scores)
            model = models[selected_reg]
        else:
            selected_reg = float(args.fixed_reg_param)
            model = fit(selected_reg)
            selection = {
                "selected_reg_param": selected_reg,
                "selection_source": "frozen_from_set_b",
            }

        write_ranker_artifacts(
            args.output,
            fill_values=fill_values,
            means=means,
            stds=stds,
            coefficients=tuple(float(value) for value in model.coefficients),
            intercept=float(model.intercept),
            reg_param=selected_reg,
            stage=args.stage,
            converged=True,
            iterations=int(model.summary.totalIterations),
            selection=selection,
            parent_paths=parse_parents(args.parent),
            scope=args.scope,
        )
        print(
            "ranker_training_ready "
            f"stage={args.stage} reg_param={selected_reg} output={args.output}",
        )
    finally:
        for frame in reversed(cached):
            try:
                frame.unpersist(blocking=False)
            except Exception:
                pass
        spark.stop()


if __name__ == "__main__":
    main()
