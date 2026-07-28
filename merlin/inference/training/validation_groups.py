"""Frozen Set-B Audio/Relation/Mixed validation-group contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..artifacts.integrity import artifact_size_bytes, sha256_path
from ..artifacts.io import write_json_atomic


VALIDATION_GROUP_VERSION = "merlin_validation_groups_v2"
VALIDATION_THRESHOLD_VERSION = "merlin_validation_groups_v1"
VALIDATION_PAIR_LAYOUT = "one_pair_nested_groups_with_selection_fold_v2"
VALIDATION_GROUP_SEED = 42
VALIDATION_QUERY_GROUPS = ("audio_dominant", "relation_dominant", "mixed")


def build_nested_validation_pairs(candidate_rows, positives):
    """Attach all eligible validation groups to one physical candidate row."""
    from pyspark.sql import functions as F

    pair_columns = ["query_track_id", "candidate_track_id"]
    eligible = positives.groupBy(
        "query_track_id", "selection_fold", "query_group"
    ).agg(
        F.count("*").cast("long").alias("eligible_positive_count")
    )
    query_folds = eligible.select(
        "query_track_id", "selection_fold"
    ).distinct()
    eligible_groups = eligible.groupBy("query_track_id").agg(
        F.sort_array(F.collect_list(F.struct(
            "query_group", "eligible_positive_count"
        ))).alias("eligible_groups")
    )
    recalled = (
        positives.select(*pair_columns, "query_group")
        .join(candidate_rows.select(*pair_columns), pair_columns, "inner")
        .dropDuplicates([*pair_columns, "query_group"])
    )
    positive_groups = recalled.groupBy(*pair_columns).agg(
        F.sort_array(F.collect_set("query_group")).alias("positive_groups")
    )
    empty_groups = F.expr("cast(array() as array<string>)")
    validation_pairs = (
        candidate_rows.join(eligible_groups, "query_track_id", "inner")
        .join(query_folds, "query_track_id", "inner")
        .join(positive_groups, pair_columns, "left")
        .withColumn(
            "validation_groups",
            F.transform(
                "eligible_groups",
                lambda group: F.struct(
                    group["query_group"].alias("query_group"),
                    F.array_contains(
                        F.coalesce(F.col("positive_groups"), empty_groups),
                        group["query_group"],
                    ).cast("int").alias("label"),
                    group["eligible_positive_count"].cast("long").alias(
                        "eligible_positive_count"
                    ),
                ),
            ),
        )
        .select(
            *pair_columns,
            "recall_sources",
            "primary_recall_sources",
            "selection_fold",
            "validation_groups",
        )
    )
    return validation_pairs, eligible, recalled


def load_selected_artist_terms(
    graph_edges_path: str | Path,
    artist_ids: Iterable[str],
    idf_values: Mapping[str, float],
) -> dict[str, set[str]]:
    """Load normalized tag sets only for selected artists."""
    from ..artifacts.io import parquet_rows

    selected = {str(artist_id) for artist_id in artist_ids}
    artist_terms: dict[str, set[str]] = {artist_id: set() for artist_id in selected}
    for artist_id, term in parquet_rows(
        graph_edges_path,
        ("src_id", "dst_id"),
        edge_type="artist_term",
        engine="pyarrow",
    ):
        artist = str(artist_id)
        normalized_term = str(term).strip().lower()
        if artist in selected and normalized_term in idf_values:
            artist_terms[artist].add(normalized_term)
    return {
        artist_id: terms for artist_id, terms in artist_terms.items() if terms
    }


def estimate_validation_scratch_gb(
    *,
    set_a_tracks: int,
    set_b_tracks: int,
    feature_dimension: int,
    unique_candidates: int,
    max_sample_pairs: int,
) -> float:
    """Estimate peak local Spark data for formal validation-group construction."""
    values = (
        set_a_tracks,
        set_b_tracks,
        feature_dimension,
        unique_candidates,
        max_sample_pairs,
    )
    if any(value <= 0 for value in values):
        raise ValueError("validation scratch estimate inputs must be positive")
    vector_checkpoint = (set_a_tracks + set_b_tracks) * feature_dimension * 8 * 1.5
    sample_sort = max_sample_pairs * 48 * 2
    nested_group_bytes = 64 + len(VALIDATION_QUERY_GROUPS) * 24
    validation_pairs = unique_candidates * nested_group_bytes * 2
    threshold_pairs = set_b_tracks * set_b_tracks * 0.15 * 48 * 2
    projected_bytes = (
        vector_checkpoint + sample_sort + validation_pairs + threshold_pairs
    )
    gib = projected_bytes / (1024**3)
    return max(4.0, math.ceil(gib * 4) / 4)


def collect_normalized_vector_matrix(
    rows: Iterable[Mapping[str, object]],
    *,
    capacity: int,
    dimension: int,
) -> tuple[dict[str, int], object]:
    """Collect valid vectors once without retaining Python copies of every array."""
    if capacity <= 0 or dimension <= 0:
        raise ValueError("vector capacity and dimension must be positive")
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("vector threshold fitting requires NumPy") from error

    positions: dict[str, int] = {}
    matrix = np.empty((capacity, dimension), dtype=np.float64)
    count = 0
    for row in rows:
        norm = float(row["pre_pca_norm"])
        if not math.isfinite(norm) or norm <= 0.0:
            continue
        track_id = str(row["track_id"])
        if not track_id or track_id in positions:
            raise ValueError("threshold vectors require unique non-empty track IDs")
        vector = np.asarray(row["pre_pca_vector"], dtype=np.float64)
        if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"threshold vector is invalid for track {track_id!r}")
        if count >= capacity:
            raise ValueError("threshold vector input exceeds its declared capacity")
        positions[track_id] = count
        matrix[count] = vector / norm
        count += 1
    if not count:
        raise ValueError("threshold vector input contains no valid rows")
    return positions, matrix[:count]


def sampled_pair_cosine_quantiles(
    pair_rows: Iterable[Mapping[str, object]],
    positions: Mapping[str, int],
    normalized_vectors: object,
    *,
    expected_pairs: int,
) -> tuple[int, float, float]:
    """Compute exact p50/p90 after deterministic ID-only pair sampling."""
    if expected_pairs <= 0:
        raise ValueError("expected pair count must be positive")
    import numpy as np

    values = np.empty(expected_pairs, dtype=np.float64)
    count = 0
    for row in pair_rows:
        if count >= expected_pairs:
            raise ValueError("sampled pair input exceeds its declared count")
        try:
            query = positions[str(row["q_track_id"])]
            candidate = positions[str(row["c_track_id"])]
        except KeyError as error:
            raise ValueError(f"sampled pair is missing a threshold vector: {error}") from error
        values[count] = np.dot(normalized_vectors[query], normalized_vectors[candidate])
        count += 1
    if count != expected_pairs or np.any(~np.isfinite(values)):
        raise ValueError("sampled pair cosine input is incomplete or invalid")
    p50, p90 = np.quantile(values, (0.5, 0.9), method="linear")
    return count, float(p50), float(p90)


def _normalized_tag_matrix(
    artist_terms: Mapping[str, Sequence[str]],
    idf_values: Mapping[str, float],
) -> tuple[tuple[str, ...], object]:
    try:
        import numpy as np
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError("sparse tag threshold search requires SciPy") from error

    artists = tuple(sorted(artist for artist, terms in artist_terms.items() if terms))
    if not artists:
        raise ValueError("tag threshold search requires tagged artists")
    terms = tuple(sorted({term for artist in artists for term in artist_terms[artist]}))
    term_positions = {term: index for index, term in enumerate(terms)}
    row_positions: list[int] = []
    column_positions: list[int] = []
    weights: list[float] = []
    for row_index, artist in enumerate(artists):
        for term in sorted(set(artist_terms[artist])):
            try:
                weight = float(idf_values[term])
            except KeyError as error:
                raise ValueError(f"tag IDF artifact is missing term {term!r}") from error
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"tag IDF weight is invalid for term {term!r}")
            row_positions.append(row_index)
            column_positions.append(term_positions[term])
            weights.append(weight)
    matrix = sparse.csr_matrix(
        (weights, (row_positions, column_positions)),
        shape=(len(artists), len(terms)),
        dtype=np.float64,
    )
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("tag vectors contain an invalid norm")
    return artists, matrix.multiply((1.0 / norms)[:, None]).tocsr()


def write_high_tag_pairs_sparse(
    artist_terms: Mapping[str, Sequence[str]],
    idf_values: Mapping[str, float],
    output_path: str | Path,
    *,
    threshold: float,
    block_size: int = 256,
) -> int:
    """Write directed artist pairs whose binary TF-IDF cosine meets a threshold."""
    if block_size <= 0 or not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("tag threshold and block size must be positive")
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("sparse tag threshold search requires NumPy and PyArrow") from error

    artists, normalized = _normalized_tag_matrix(artist_terms, idf_values)
    schema = pa.schema((
        pa.field("q_artist_id", pa.string(), nullable=False),
        pa.field("c_artist_id", pa.string(), nullable=False),
    ))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, schema, compression="zstd", use_dictionary=True)
    count = 0
    try:
        for start in range(0, len(artists), block_size):
            similarities = (normalized[start : start + block_size] @ normalized.T).tocoo()
            query_positions = similarities.row + start
            selected = (
                (similarities.data >= threshold)
                & (query_positions != similarities.col)
            )
            query_positions = query_positions[selected]
            candidate_positions = similarities.col[selected]
            if not len(query_positions):
                continue
            order = np.lexsort((candidate_positions, query_positions))
            query_positions = query_positions[order]
            candidate_positions = candidate_positions[order]
            writer.write_table(pa.Table.from_arrays(
                (
                    pa.array([artists[index] for index in query_positions]),
                    pa.array([artists[index] for index in candidate_positions]),
                ),
                schema=schema,
            ))
            count += len(query_positions)
    finally:
        writer.close()
    temporary.replace(output)
    return count


def write_audio_threshold_pairs_numpy(
    query_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    threshold: float,
    block_size: int = 256,
) -> int:
    """Write blocked exact cosine-threshold pairs without a Spark cross product."""
    if block_size <= 0 or not math.isfinite(threshold):
        raise ValueError("audio threshold and block size must be valid")
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("NumPy audio threshold search requires numpy and pyarrow") from error

    queries = sorted(query_rows, key=lambda row: str(row["query_track_id"]))
    candidates = sorted(candidate_rows, key=lambda row: str(row["candidate_track_id"]))
    if not queries or not candidates:
        raise ValueError("audio threshold search requires query and candidate vectors")

    def normalized_matrix(rows, vector_key: str, norm_key: str):
        matrix = np.asarray([row[vector_key] for row in rows], dtype=np.float64)
        norms = np.asarray([row[norm_key] for row in rows], dtype=np.float64)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("audio threshold vectors are invalid")
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("audio threshold vector norms are invalid")
        return matrix / norms[:, None]

    query_matrix = normalized_matrix(queries, "q_vector", "q_norm")
    candidate_matrix = normalized_matrix(candidates, "c_vector", "c_norm")
    query_tracks = np.asarray(
        [str(row["query_track_id"]) for row in queries], dtype=object
    )
    query_songs = np.asarray([row.get("q_song_id") for row in queries], dtype=object)
    query_artists = np.asarray(
        [str(row["q_artist_id"]) for row in queries], dtype=object
    )
    query_releases = np.asarray([row["q_release_id"] for row in queries], dtype=object)
    candidate_tracks = np.asarray(
        [str(row["candidate_track_id"]) for row in candidates], dtype=object
    )
    candidate_songs = np.asarray([row.get("c_song_id") for row in candidates], dtype=object)
    candidate_artists = np.asarray(
        [str(row["c_artist_id"]) for row in candidates], dtype=object
    )
    candidate_releases = np.asarray([row["c_release_id"] for row in candidates], dtype=object)
    symmetric = (
        np.array_equal(query_tracks, candidate_tracks)
        and np.array_equal(query_songs, candidate_songs)
        and np.array_equal(query_artists, candidate_artists)
        and np.array_equal(query_releases, candidate_releases)
        and np.array_equal(query_matrix, candidate_matrix)
    )

    schema = pa.schema((
        pa.field("query_track_id", pa.string(), nullable=False),
        pa.field("candidate_track_id", pa.string(), nullable=False),
        pa.field("q_artist_id", pa.string(), nullable=False),
        pa.field("c_artist_id", pa.string(), nullable=False),
    ))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, schema, compression="zstd", use_dictionary=True)
    count = 0
    try:
        for start in range(0, len(queries), block_size):
            stop = min(start + block_size, len(queries))
            candidate_start = start if symmetric else 0
            block_tracks = query_tracks[start:stop]
            block_songs = query_songs[start:stop]
            block_artists = query_artists[start:stop]
            block_releases = query_releases[start:stop]
            compared_tracks = candidate_tracks[candidate_start:]
            compared_songs = candidate_songs[candidate_start:]
            compared_artists = candidate_artists[candidate_start:]
            compared_releases = candidate_releases[candidate_start:]
            mask = (
                query_matrix[start:stop] @ candidate_matrix[candidate_start:].T
                >= threshold
            )
            mask &= block_tracks[:, None] != compared_tracks[None, :]
            mask &= block_artists[:, None] != compared_artists[None, :]
            mask &= block_releases[:, None] != compared_releases[None, :]
            same_song = (
                (block_songs[:, None] != None)  # noqa: E711
                & (compared_songs[None, :] != None)  # noqa: E711
                & (block_songs[:, None] == compared_songs[None, :])
            )
            mask &= ~same_song
            query_positions, candidate_positions = np.nonzero(mask)
            query_positions += start
            candidate_positions += candidate_start
            if symmetric:
                upper = candidate_positions > query_positions
                query_positions = query_positions[upper]
                candidate_positions = candidate_positions[upper]
            if not len(query_positions):
                continue
            if symmetric:
                query_positions, candidate_positions = (
                    np.concatenate((query_positions, candidate_positions)),
                    np.concatenate((candidate_positions, query_positions)),
                )
            writer.write_table(pa.Table.from_arrays(
                (
                    pa.array(query_tracks[query_positions].tolist(), type=pa.string()),
                    pa.array(candidate_tracks[candidate_positions].tolist(), type=pa.string()),
                    pa.array(query_artists[query_positions].tolist(), type=pa.string()),
                    pa.array(candidate_artists[candidate_positions].tolist(), type=pa.string()),
                ),
                schema=schema,
            ))
            count += len(query_positions)
    finally:
        writer.close()
    temporary.replace(output)
    return count


def classify_validation_pair(
    pair: Mapping[str, object],
    *,
    acoustic_p50: float,
    acoustic_p90: float,
    tag_positive_threshold: float,
) -> tuple[bool, bool]:
    """Return Audio-dominant and Relation-dominant membership for one pair."""
    if not all(math.isfinite(value) for value in (
        acoustic_p50,
        acoustic_p90,
        tag_positive_threshold,
    )):
        raise ValueError("validation-group thresholds must be finite")
    if not acoustic_p50 <= acoustic_p90:
        raise ValueError("pre-PCA acoustic thresholds must satisfy p50 <= p90")
    acoustic = float(pair["pre_pca_cosine"])
    if not math.isfinite(acoustic):
        raise ValueError("pre-PCA acoustic cosine must be finite")
    tag_value = pair.get("tag_tfidf_cosine")
    tag = None if tag_value is None else float(tag_value)
    if tag is not None and not math.isfinite(tag):
        raise ValueError("artist-term cosine must be finite when present")

    same_song = bool(pair.get("same_song", False))
    same_artist = bool(pair.get("same_artist", False))
    same_release = bool(pair.get("same_release", False))
    has_artist_pair = bool(pair.get("has_artist_pair", False))
    has_release_pair = bool(pair.get("has_release_pair", False))
    directed_artist_similarity = bool(pair.get("directed_artist_similarity", False))
    if same_song:
        return False, False

    audio_dominant = (
        acoustic >= acoustic_p90
        and has_artist_pair
        and not same_artist
        and has_release_pair
        and not same_release
        and tag is not None
        and tag < tag_positive_threshold
    )
    relation_signal = (
        same_artist
        or same_release
        or directed_artist_similarity
        or (tag is not None and tag >= tag_positive_threshold)
    )
    relation_dominant = acoustic < acoustic_p50 and relation_signal
    return audio_dominant, relation_dominant


def mixed_positive_ids(
    query_track_id: str,
    audio_positive_ids: set[str],
    relation_positive_ids: set[str],
    *,
    seed: int = VALIDATION_GROUP_SEED,
) -> tuple[str, ...]:
    """Select equal Audio/Relation counts without backfilling a short side."""
    import hashlib

    if not query_track_id:
        raise ValueError("mixed validation query ID must not be empty")
    overlap = audio_positive_ids & relation_positive_ids
    if overlap:
        raise ValueError("Audio- and Relation-dominant positives must be disjoint")
    count = min(len(audio_positive_ids), len(relation_positive_ids))

    def ordered(values: set[str], source: str) -> list[str]:
        return sorted(
            values,
            key=lambda candidate_id: (
                hashlib.sha256(
                    f"{seed}\0{query_track_id}\0{source}\0{candidate_id}".encode("utf-8")
                ).hexdigest(),
                candidate_id,
            ),
        )

    selected = ordered(audio_positive_ids, "audio")[:count]
    selected.extend(ordered(relation_positive_ids, "relation")[:count])
    return tuple(selected)


def write_validation_group_manifest(
    manifest_path: str | Path,
    *,
    thresholds_path: str | Path,
    positives_path: str | Path,
    validation_pairs_path: str | Path,
    parent_paths: Mapping[str, str | Path],
    scope: str,
    threshold_sample_count: int,
    pair_count: int,
    group_row_count: int,
    group_stats: Mapping[str, Mapping[str, int | float]],
    apply_split: str = "set_b",
) -> dict[str, object]:
    if scope not in {"formal", "smoke"}:
        raise ValueError("validation-group scope must be formal or smoke")
    if set(group_stats) != set(VALIDATION_QUERY_GROUPS):
        raise ValueError("validation-group statistics must cover all frozen groups")
    if threshold_sample_count <= 0:
        raise ValueError("validation-group threshold sample must not be empty")
    if pair_count <= 0 or group_row_count < pair_count:
        raise ValueError("validation-pair layout counts are invalid")
    if apply_split not in {"set_b", "set_c"}:
        raise ValueError("validation groups may only be applied to Set B or Set C")
    manifest = {
        "artifact_type": f"{apply_split}_validation_groups",
        "artifact_version": VALIDATION_GROUP_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "fit_split": "set_a",
        "apply_split": apply_split,
        "seed": VALIDATION_GROUP_SEED,
        "threshold_sample_count": int(threshold_sample_count),
        "validation_pair_layout": VALIDATION_PAIR_LAYOUT,
        "validation_pair_ordering": ["query_track_id", "candidate_track_id"],
        "validation_pair_count": int(pair_count),
        "validation_group_row_count": int(group_row_count),
        "query_groups": list(VALIDATION_QUERY_GROUPS),
        "group_stats": {name: dict(group_stats[name]) for name in VALIDATION_QUERY_GROUPS},
        "thresholds_file": Path(thresholds_path).name,
        "thresholds_sha256": sha256_path(thresholds_path),
        "positives_path": Path(positives_path).name,
        "positives_sha256": sha256_path(positives_path),
        "positives_size_bytes": artifact_size_bytes(positives_path),
        "validation_pairs_path": Path(validation_pairs_path).name,
        "validation_pairs_sha256": sha256_path(validation_pairs_path),
        "validation_pairs_size_bytes": artifact_size_bytes(validation_pairs_path),
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_validation_group_manifest(
    manifest_path: str | Path,
    *,
    thresholds_path: str | Path,
    positives_path: str | Path,
    validation_pairs_path: str | Path,
    expected_scope: str,
    expected_parent_hashes: Mapping[str, str] | None = None,
    expected_apply_split: str = "set_b",
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if expected_apply_split not in {"set_b", "set_c"}:
        raise ValueError("validation-group expected split is invalid")
    if manifest.get("artifact_type") != f"{expected_apply_split}_validation_groups":
        raise ValueError("validation-group artifact type mismatch")
    if manifest.get("artifact_version") != VALIDATION_GROUP_VERSION:
        raise ValueError("validation-group artifact version mismatch")
    if manifest.get("scope") != expected_scope:
        raise ValueError("validation-group scope mismatch")
    if (
        manifest.get("fit_split") != "set_a"
        or manifest.get("apply_split") != expected_apply_split
    ):
        raise ValueError("validation-group split boundary mismatch")
    if manifest.get("query_groups") != list(VALIDATION_QUERY_GROUPS):
        raise ValueError("validation-group names or order mismatch")
    if manifest.get("validation_pair_layout") != VALIDATION_PAIR_LAYOUT:
        raise ValueError("validation-pair row layout mismatch")
    if manifest.get("validation_pair_ordering") != [
        "query_track_id", "candidate_track_id"
    ]:
        raise ValueError("validation-pair ordering mismatch")
    pair_count = int(manifest.get("validation_pair_count", 0))
    group_row_count = int(manifest.get("validation_group_row_count", 0))
    if pair_count <= 0 or group_row_count < pair_count:
        raise ValueError("validation-pair manifest counts are invalid")
    with Path(thresholds_path).open("r", encoding="utf-8") as stream:
        thresholds = json.load(stream)
    if thresholds.get("artifact_type") != "set_b_validation_thresholds":
        raise ValueError("validation-group threshold artifact type mismatch")
    if thresholds.get("artifact_version") != VALIDATION_GROUP_VERSION:
        raise ValueError("validation-group threshold artifact version mismatch")
    if thresholds.get("fit_split") != "set_a":
        raise ValueError("validation-group threshold fit split mismatch")
    acoustic_p50 = float(thresholds.get("pre_pca_acoustic_cosine_p50", math.nan))
    acoustic_p90 = float(thresholds.get("pre_pca_acoustic_cosine_p90", math.nan))
    if not math.isfinite(acoustic_p50) or not math.isfinite(acoustic_p90):
        raise ValueError("validation-group acoustic thresholds are not finite")
    if acoustic_p50 > acoustic_p90:
        raise ValueError("validation-group acoustic thresholds are reversed")
    tag_threshold = float(thresholds.get("tag_positive_threshold", math.nan))
    if not math.isfinite(tag_threshold):
        raise ValueError("validation-group tag threshold is not finite")
    artifacts = (
        ("thresholds", Path(thresholds_path)),
        ("positives", Path(positives_path)),
        ("validation_pairs", Path(validation_pairs_path)),
    )
    for name, path in artifacts:
        path_key = "thresholds_file" if name == "thresholds" else f"{name}_path"
        if manifest.get(path_key) != path.name:
            raise ValueError(f"validation-group {name} path mismatch")
        if manifest.get(f"{name}_sha256") != sha256_path(path):
            raise ValueError(f"validation-group {name} hash mismatch")
    parents = manifest.get("parent_hashes")
    if not isinstance(parents, dict):
        raise ValueError("validation-group parent hashes are missing")
    for name, expected_hash in (expected_parent_hashes or {}).items():
        if parents.get(name) != expected_hash:
            raise ValueError(f"validation-group parent hash mismatch: {name}")
    stats = manifest.get("group_stats")
    if not isinstance(stats, dict) or set(stats) != set(VALIDATION_QUERY_GROUPS):
        raise ValueError("validation-group statistics are incomplete")
    return manifest
