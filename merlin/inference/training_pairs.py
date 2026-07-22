        track_id = str(candidate["track_id"])
        recall_sources = frozenset(str(value) for value in candidate["recall_sources"])
        if track_id == query_id:
            rejection_counts["query_self"] += 1
        elif track_id not in allowed_tracks:
            rejection_counts["outside_universe"] += 1
        elif track_id in seen_candidates:
            rejection_counts["duplicate_pair"] += 1
        elif same_song(query_id, track_id):
            rejection_counts["same_song"] += 1
        elif track_id in positive_ids:
            rejection_counts["known_positive"] += 1
        else:
            seen_candidates.add(track_id)
            predicate_candidates.append((track_id, recall_sources))
    predicate_results = (
        is_positive_batch(
            query_id,
            [track_id for track_id, _sources in predicate_candidates],
        )
        if is_positive_batch is not None
        else [is_positive(query_id, track_id) for track_id, _sources in predicate_candidates]
    )
    if len(predicate_results) != len(predicate_candidates):
        raise ValueError("batch positive predicate returned the wrong number of results")
    for candidate, positive in zip(predicate_candidates, predicate_results, strict=True):
        if positive:
            rejection_counts["known_positive"] += 1
        else:
            eligible_candidates.append(candidate)
    eligible_candidates.sort(
        key=lambda item: _pair_hash(query_id, item[0], "candidate_aware")
    )
    candidate_selected = eligible_candidates[:candidate_target]
    rejected = positive_ids | {track_id for track_id, _sources in candidate_selected}
    random_target = negative_target - len(candidate_selected)
    random_selected = _random_negatives(
        query_id,
        random_universe,
        random_target,
        rejected,
        same_song,
        is_positive,
        rejection_counts,
    )
    if len(candidate_selected) + len(random_selected) != negative_target:
        raise ValueError(f"negative sampling shortage for query {query_id}")

    rows = [
        {
            "query_track_id": query_id,
            "candidate_track_id": track_id,
            "label": 1,
            "positive_sources": sorted(sources),
            "negative_source": None,
            "recall_sources": [],
        }
        for track_id, sources in sorted(selected_positives.items())
    ]
    rows.extend(
        {
            "query_track_id": query_id,
            "candidate_track_id": track_id,
            "label": 0,
            "positive_sources": [],
            "negative_source": "candidate_aware",
            "recall_sources": sorted(sources),
        }
        for track_id, sources in candidate_selected
    )
    rows.extend(
        {
            "query_track_id": query_id,
            "candidate_track_id": track_id,
            "label": 0,
            "positive_sources": [],
            "negative_source": "random",
            "recall_sources": [],
        }
        for track_id in random_selected
    )
    rows.sort(
        key=lambda row: (
            -int(row["label"]),
            _pair_hash(query_id, str(row["candidate_track_id"]), "output"),
        )
    )
    return rows, {
        "positive_count": len(positive_ids),
        "negative_count": negative_target,
        "candidate_aware_count": len(candidate_selected),
        "random_count": len(random_selected),
        "candidate_shortage": candidate_target - len(candidate_selected),
        "negative_shortage": 0,
        "rejections": dict(sorted(rejection_counts.items())),
    }


def write_training_pair_artifacts(
    candidate_pool_path: str | Path,
    positives: Mapping[str, Mapping[str, frozenset[str]]],
    assignments: Mapping[str, str],
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    stage: str,
    same_song: SameSong,
    is_positive: IsPositive,
    parent_paths: Mapping[str, str | Path],
    scope: str,
    is_positive_batch: IsPositiveBatch | None = None,
) -> dict[str, object]:
    if scope not in {"formal", "smoke"}:
        raise ValueError("training-pair scope must be formal or smoke")
    allowed = allowed_training_tracks(assignments, stage)
    universe = tuple(sorted(allowed))
    totals: Counter[str] = Counter()
    rejection_totals: Counter[str] = Counter()
    query_count = 0

    def pair_rows() -> Iterator[dict[str, object]]:
        nonlocal query_count
        for record in iter_candidate_pool(candidate_pool_path):
            query_id = str(record["query_track_id"])
            if query_id not in positives or query_id not in allowed:
                continue
            rows, audit = construct_query_pairs(
                query_id,
                positives[query_id],
                record["candidates"],
                allowed,
                universe,
                same_song,
                is_positive,
                is_positive_batch,
            )
            if not rows:
                continue
            query_count += 1
            for key in (
                "positive_count",
                "negative_count",
                "candidate_aware_count",
                "random_count",
                "candidate_shortage",
            ):
                totals[key] += int(audit[key])
            rejection_totals.update(audit["rejections"])
            yield from rows

    output = Path(output_path)
    parquet_schema = None
    if output.suffix == ".parquet":
        import pyarrow as pa

        parquet_schema = pa.schema((
            pa.field("query_track_id", pa.string(), nullable=False),
            pa.field("candidate_track_id", pa.string(), nullable=False),
            pa.field("label", pa.int64(), nullable=False),
            pa.field("positive_sources", pa.list_(pa.string()), nullable=False),
            pa.field("negative_source", pa.string()),
            pa.field("recall_sources", pa.list_(pa.string()), nullable=False),
        ))
    pair_count = write_row_artifact(pair_rows(), output, parquet_schema=parquet_schema)
    if query_count == 0 or pair_count == 0:
        raise ValueError("training-pair artifact has no eligible query pairs")
    if totals["negative_count"] != NEGATIVE_RATIO * totals["positive_count"]:
        raise ValueError("training-pair artifact violates the 1:3 ratio")
    manifest = {
        "artifact_type": "ranker_training_pairs",
        "artifact_version": TRAINING_PAIR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "stage": stage,
        "seed": PAIR_SEED,
        "negative_ratio": NEGATIVE_RATIO,
        "candidate_aware_target_fraction": CANDIDATE_AWARE_FRACTION,
        "query_count": query_count,
        "pair_count": pair_count,
        "counts": dict(sorted(totals.items())),
        "actual_candidate_aware_fraction": (
            totals["candidate_aware_count"] / totals["negative_count"]
        ),
        "rejection_counts": dict(sorted(rejection_totals.items())),
        "pairs_file": output.name,
        "storage_format": "parquet" if output.suffix == ".parquet" else "jsonl_gzip",
        "pairs_sha256": sha256_path(output),
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_training_pair_manifest(
    manifest_path: str | Path,
    pairs_path: str | Path,
    *,
    expected_scope: str,
    expected_stage: str,
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != "ranker_training_pairs":
        raise ValueError("training-pair artifact type mismatch")
    if manifest.get("artifact_version") != TRAINING_PAIR_VERSION:
        raise ValueError("training-pair artifact version mismatch")
    if manifest.get("scope") != expected_scope:
        raise ValueError("training-pair scope mismatch")
    if manifest.get("stage") != expected_stage:
        raise ValueError("training-pair stage mismatch")
    pairs = Path(pairs_path)
    if manifest.get("pairs_file") != pairs.name:
        raise ValueError("training-pair output path mismatch")
    if manifest.get("pairs_sha256") != sha256_path(pairs):
        raise ValueError("training-pair output hash mismatch")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("training-pair counts are missing")
    if int(counts.get("negative_count", -1)) != NEGATIVE_RATIO * int(
        counts.get("positive_count", -1)
    ):
        raise ValueError("training-pair manifest violates the 1:3 ratio")
    return manifest
