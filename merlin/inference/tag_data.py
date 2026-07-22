"""Runtime data generation for shared-tag candidate recall."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TagData:
    """Compact song ownership and bidirectional artist-tag mappings."""

    track_to_artist: Mapping[str, str]
    artist_tracks: Mapping[str, Sequence[str]]
    artist_terms: Mapping[str, frozenset[str]]
    term_artists: Mapping[str, frozenset[str]]


def artist_tag_cosine(
    data: TagData,
    left_artist: str,
    right_artist: str,
    idf_values: Mapping[str, float] | None = None,
) -> float | None:
    """Compute exact binary TF-IDF cosine for one artist pair."""
    left = data.artist_terms.get(left_artist)
    right = data.artist_terms.get(right_artist)
    if not left or not right or not data.artist_terms:
        return None

    def idf(term: str) -> float:
        if idf_values is not None:
            try:
                return float(idf_values[term])
            except KeyError as error:
                raise ValueError(f"tag IDF artifact is missing term {term!r}") from error
        frequency = len(data.term_artists.get(term, ()))
        return math.log((1.0 + len(data.artist_terms)) / (1.0 + frequency)) + 1.0

    numerator = sum(idf(term) ** 2 for term in left & right)
    left_norm = math.sqrt(sum(idf(term) ** 2 for term in left))
    right_norm = math.sqrt(sum(idf(term) ** 2 for term in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return numerator / (left_norm * right_norm)


def build_tag_data(
    songs: Iterable[tuple[str, str]],
    terms: Iterable[tuple[str, str]],
) -> TagData:
    """Build duplicate-free mappings from projected metadata rows."""
    track_to_artist: dict[str, str] = {}
    artist_tracks: dict[str, list[str]] = defaultdict(list)
    seen_tracks: set[str] = set()
    for track_id, artist_id in songs:
        if not track_id or not artist_id:
            continue
        previous = track_to_artist.setdefault(track_id, artist_id)
        if previous != artist_id:
            raise ValueError(f"track {track_id!r} has multiple artists")
        if track_id not in seen_tracks:
            artist_tracks[artist_id].append(track_id)
            seen_tracks.add(track_id)

    artist_terms: dict[str, set[str]] = defaultdict(set)
    term_artists: dict[str, set[str]] = defaultdict(set)
    for artist_id, term in terms:
        if not artist_id or not term:
            continue
        normalized = term.strip().lower()
        if normalized:
            artist_terms[artist_id].add(normalized)
            term_artists[normalized].add(artist_id)

    return TagData(
        track_to_artist=track_to_artist,
        artist_tracks=dict(artist_tracks),
        artist_terms={key: frozenset(value) for key, value in artist_terms.items()},
        term_artists={key: frozenset(value) for key, value in term_artists.items()},
    )


def load_tag_data(
    songs_metadata_path: str | Path,
    graph_edges_path: str | Path,
) -> TagData:
    """Load metadata and canonical artist-term graph edges."""
    songs = _parquet_rows(songs_metadata_path, ("track_id", "artist_id"))
    terms = _graph_edge_rows(graph_edges_path, "artist_term")
    return build_tag_data(songs, terms)


def _graph_edge_rows(path: str | Path, edge_type: str) -> Iterable[tuple[str, str]]:
    try:
        import pyarrow.dataset as ds
    except ImportError as error:
        raise RuntimeError("loading Tag graph edges requires pyarrow") from error
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    scanner = dataset.scanner(
        columns=["src_id", "dst_id"],
        filter=ds.field("edge_type") == edge_type,
    )
    for batch in scanner.to_batches():
        yield from zip(batch.column(0).to_pylist(), batch.column(1).to_pylist())


def _parquet_rows(
    path: str | Path,
    columns: tuple[str, str],
) -> Iterable[tuple[str, str]]:
    try:
        import pyarrow.dataset as ds
    except ImportError as error:
        raise RuntimeError("loading tag Parquet data requires pyarrow") from error

    dataset = ds.dataset(str(path), format="parquet")
    for batch in dataset.to_batches(columns=list(columns)):
        left = batch.column(0).to_pylist()
        right = batch.column(1).to_pylist()
        yield from zip(left, right)


def find_similar_artists(
    data: TagData,
    artist_id: str,
    top_k: int,
    *,
    max_term_artists: int = 5_000,
    idf_values: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Rank artists by cosine similarity over binary TF-IDF tag vectors."""
    if top_k <= 0 or max_term_artists <= 0:
        raise ValueError("tag neighbor limits must be positive")
    query_terms = data.artist_terms.get(artist_id, frozenset())
    artist_count = len(data.artist_terms)
    if not query_terms or artist_count == 0:
        return []

    def idf(term: str) -> float:
        if idf_values is not None:
            try:
                return float(idf_values[term])
            except KeyError as error:
                raise ValueError(f"tag IDF artifact is missing term {term!r}") from error
        frequency = len(data.term_artists.get(term, ()))
        return math.log((1.0 + artist_count) / (1.0 + frequency)) + 1.0

    usable_terms = [
        term
        for term in query_terms
        if len(data.term_artists.get(term, ())) <= max_term_artists
    ]
    query_norm = math.sqrt(sum(idf(term) ** 2 for term in query_terms))
    numerators: dict[str, float] = defaultdict(float)
    for term in usable_terms:
        weight = idf(term) ** 2
        for candidate in data.term_artists[term]:
            if candidate != artist_id:
                numerators[candidate] += weight

    scored = []
    for candidate, numerator in numerators.items():
        candidate_norm = math.sqrt(
            sum(idf(term) ** 2 for term in data.artist_terms[candidate])
        )
        if candidate_norm > 0.0:
            scored.append((candidate, numerator / (query_norm * candidate_norm)))
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


def load_tag_idf(path: str | Path) -> dict[str, float]:
    """Load and validate Person A's frozen artist-term IDF artifact."""
    with Path(path).open("r", encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("artifact_type") != TAG_IDF_ARTIFACT_TYPE:
        raise ValueError("unsupported tag IDF artifact type")
    if artifact.get("manifest_version") != TAG_IDF_MANIFEST_VERSION:
        raise ValueError("unsupported tag IDF manifest version")
    if artifact.get("formula") != TAG_IDF_FORMULA:
        raise ValueError("tag IDF formula mismatch")
    values = {str(term): float(value) for term, value in artifact["values"].items()}
    if not values:
        raise ValueError("tag IDF artifact must not be empty")
    if int(artifact.get("term_count", -1)) != len(values):
        raise ValueError("tag IDF term count mismatch")
    if int(artifact.get("artist_count", -1)) <= 0:
        raise ValueError("tag IDF artist count must be positive")
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("tag IDF values must be finite and positive")
    if expected_graph_edges_path is not None:
        artist_term_path = Path(expected_graph_edges_path) / "edge_type=artist_term"
        source = artifact.get("source")
        if not isinstance(source, dict):
            raise ValueError("tag IDF source lineage is missing")
        expected_hash = source.get("artist_term_sha256")
        if expected_hash != sha256_path(artist_term_path):
            raise ValueError("tag IDF artist-term lineage mismatch")
    return values
