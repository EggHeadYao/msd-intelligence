            and float(score) >= tag_threshold
        ),
        key=lambda item: (-item[1], item[0]),
    )
    source_lists = {
        "same_artist": same_artist,
        "tag_derived": [track_id for track_id, _score in tags],
        "audio_derived": [track_id for track_id, _score in audio],
    }
    for source, tracks in source_lists.items():
        for track_id in tracks:
            provenance.setdefault(track_id, set()).add(source)

    positions = {source: 0 for source in POSITIVE_SOURCES}
    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < limit:
        progressed = False
        for source in POSITIVE_SOURCES:
            tracks = source_lists[source]
            while positions[source] < len(tracks):
                track_id = tracks[positions[source]]
                positions[source] += 1
                if track_id in selected_set:
                    continue
                selected.append(track_id)
                selected_set.add(track_id)
                progressed = True
                break
            if len(selected) == limit:
                break
        if not progressed:
            break
    return [
        {"track_id": track_id, "positive_sources": sorted(provenance[track_id])}
        for track_id in selected
    ]


def write_weak_positive_artifacts(
    records: Iterable[Mapping[str, object]],
    positives_path: str | Path,
    manifest_path: str | Path,
    thresholds_path: str | Path,
    thresholds: Mapping[str, object],
    *,
    parent_paths: Mapping[str, str | Path],
    scope: str,
) -> dict[str, object]:
    if scope not in {"formal", "smoke"}:
        raise ValueError("weak-positive scope must be formal or smoke")
    write_json_atomic(thresholds, thresholds_path)
    query_count = 0
    positive_count = 0
    source_counts: Counter[str] = Counter()

    def counted() -> Iterator[Mapping[str, object]]:
        nonlocal query_count, positive_count
        for record in records:
            positives = record.get("positives")
            if not isinstance(positives, list):
                raise ValueError("weak-positive record is missing positives")
            query_count += 1
            positive_count += len(positives)
            for positive in positives:
                for source in positive["positive_sources"]:
                    source_counts[str(source)] += 1
            yield record

    output = Path(positives_path)
    parquet_schema = None
    if output.suffix == ".parquet":
        import pyarrow as pa

        parquet_schema = pa.schema((
            pa.field("query_track_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("positives", pa.list_(pa.struct((
                pa.field("track_id", pa.string(), nullable=False),
                pa.field("positive_sources", pa.list_(pa.string()), nullable=False),
            ))), nullable=False),
        ))
    write_row_artifact(counted(), output, parquet_schema=parquet_schema)
    manifest = {
        "artifact_type": "weak_positives",
        "artifact_version": WEAK_LABEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "query_count": query_count,
        "positive_count": positive_count,
        "max_positives_per_query": MAX_POSITIVES_PER_QUERY,
        "source_counts": dict(sorted(source_counts.items())),
        "positives_file": output.name,
        "storage_format": "parquet" if output.suffix == ".parquet" else "jsonl_gzip",
        "positives_sha256": sha256_path(output),
        "thresholds_file": Path(thresholds_path).name,
        "thresholds_sha256": sha256_path(thresholds_path),
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_weak_positives(path: str | Path) -> dict[str, dict[str, frozenset[str]]]:
    result: dict[str, dict[str, frozenset[str]]] = {}
    for row in read_row_artifact(path):
        query_id = str(row["query_track_id"])
        if query_id in result:
            raise ValueError("weak positives contain a duplicate query")
        positives: dict[str, frozenset[str]] = {}
        for positive in row["positives"]:
            track_id = str(positive["track_id"])
            sources = frozenset(str(value) for value in positive["positive_sources"])
            if not sources or not sources.issubset(POSITIVE_SOURCES):
                raise ValueError("weak positive contains invalid provenance")
            if track_id in positives:
                raise ValueError("weak positive record contains a duplicate track")
            positives[track_id] = sources
        result[query_id] = positives
    return result


def load_weak_positive_manifest(
    manifest_path: str | Path,
    positives_path: str | Path,
    thresholds_path: str | Path,
    *,
    expected_scope: str,
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != "weak_positives":
        raise ValueError("weak-positive artifact type mismatch")
    if manifest.get("artifact_version") != WEAK_LABEL_VERSION:
        raise ValueError("weak-positive artifact version mismatch")
    if manifest.get("scope") != expected_scope:
        raise ValueError("weak-positive scope mismatch")
    positives = Path(positives_path)
    thresholds = Path(thresholds_path)
    if manifest.get("positives_file") != positives.name:
        raise ValueError("weak-positive output path mismatch")
    if manifest.get("positives_sha256") != sha256_path(positives):
        raise ValueError("weak-positive output hash mismatch")
    if manifest.get("thresholds_file") != thresholds.name:
        raise ValueError("weak-positive threshold path mismatch")
    if manifest.get("thresholds_sha256") != sha256_path(thresholds):
        raise ValueError("weak-positive threshold hash mismatch")
    return manifest
