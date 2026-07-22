    spark = SparkSession.builder.appName("MerlinRankerTraining").getOrCreate()
    cached = []
    try:
        def read_rows(path: Path):
            return (
                spark.read.parquet(str(path))
                if path.suffix == ".parquet"
                else spark.read.json(str(path))
            )

        train = read_rows(args.train_features).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        cached.append(train)
        statistics = train.agg(
            F.count("*").alias("row_count"),
            *(
                F.percentile_approx(name, 0.5, 1_000_000).alias(f"median_{name}")
                for name in FILL_FEATURES
            ),
        ).first()
        if int(statistics["row_count"]) == 0:
            raise ValueError("Ranker training features are empty")
        fill_values = {}
        for name in FILL_FEATURES:
            value = statistics[f"median_{name}"]
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"Set-A fill statistic is invalid: {name}")
            fill_values[name] = float(value)

        def materialize(frame: Any) -> Any:
            result = frame
            for name, value in fill_values.items():
                result = result.withColumn(name, F.coalesce(F.col(name), F.lit(value)))
            result = result.withColumn(
                "audio_tag_interaction",
                F.col("cos_audio") * F.col("tag_tfidf_cosine"),
            ).withColumn(
                "graph_bfs_interaction",
                F.col("cos_graph") * F.col("bfs_score"),
            )
            return result

        assembler = VectorAssembler(
            inputCols=list(RANKER_V2_FEATURES),
            outputCol="unscaled_features",
            handleInvalid="error",
        )
        assembled_train = assembler.transform(materialize(train))
        scaler = StandardScaler(
            inputCol="unscaled_features",
            outputCol="features",
            withMean=True,
            withStd=True,
        ).fit(assembled_train)
        scaled_train = scaler.transform(assembled_train).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        cached.append(scaled_train)
        means = tuple(float(value) for value in scaler.mean)
        stds = tuple(float(value) for value in scaler.std)

        def fit(reg_param: float) -> Any:
            model = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                elasticNetParam=0.0,
                regParam=reg_param,
                maxIter=100,
                tol=1e-6,
                fitIntercept=True,
                standardization=True,
            ).fit(scaled_train)
            if int(model.summary.totalIterations) >= 100:
                raise ValueError(f"LR did not converge for regParam={reg_param}")
            return model

        selection: dict[str, object]
        if args.stage == "tuning":
            validation = read_rows(args.validation_features).persist(
                StorageLevel.MEMORY_AND_DISK
            )
            cached.append(validation)
            invalid_groups = validation.where(
                ~F.col("query_group").isin(*QUERY_GROUPS)
                | F.col("eligible_positive_count").isNull()
                | (F.col("eligible_positive_count") <= 0)
            ).limit(1).count()
            if invalid_groups:
                raise ValueError("Set-B validation contains an invalid group or denominator")
            scaled_validation = scaler.transform(
                assembler.transform(materialize(validation))
            ).persist(StorageLevel.MEMORY_AND_DISK)
            cached.append(scaled_validation)
            models = {}
            for reg_param in REG_PARAMS:
                model = fit(reg_param)
                models[reg_param] = model
            feature_array = vector_to_array("features")
            score_structs = []
            for reg_param in REG_PARAMS:
                fitted = models[reg_param]
                coefficient_array = F.array(
                    *(F.lit(float(value)) for value in fitted.coefficients)
                )
                margin = F.aggregate(
                    F.zip_with(
                        feature_array,
                        coefficient_array,
                        lambda feature, coefficient: feature * coefficient,
                    ),
                    F.lit(float(fitted.intercept)),
                    lambda total, value: total + value,
                )
                score_structs.append(F.struct(
                    F.lit(float(reg_param)).alias("reg_param"),
                    margin.alias("margin"),
                ))
            predictions = (
                scaled_validation.withColumn(
                    "model_score", F.explode(F.array(*score_structs))
                )
                .select("*", "model_score.*")
                .drop("model_score")
            )
            ranking = Window.partitionBy(
                "reg_param", "query_track_id", "query_group"
            ).orderBy(F.desc("margin"), F.asc("candidate_track_id"))
            query_window = Window.partitionBy(
                "reg_param", "query_track_id", "query_group"
            )
            per_query = (
                predictions.withColumn("rank", F.row_number().over(ranking))
                .withColumn(
                    "positive_count",
                    F.max("eligible_positive_count").over(query_window),
                )
                .withColumn(
                    "gain",
                    F.when(
                        (F.col("rank") <= 20) & (F.col("label") == 1),
                        1.0 / F.log2(F.col("rank") + 1.0),
                    ).otherwise(0.0),
                )
                .groupBy("reg_param", "query_track_id", "query_group")
                .agg(
                    F.sum("gain").alias("dcg"),
                    F.max("positive_count").alias("positive_count"),
                )
                .withColumn(
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
