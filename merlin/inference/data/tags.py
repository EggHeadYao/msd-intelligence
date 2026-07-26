"""Runtime data generation for shared-tag candidate recall."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..artifacts.integrity import sha256_path
from ..artifacts.io import parquet_rows


TAG_IDF_ARTIFACT_TYPE = "artist_term_idf"
TAG_IDF_MANIFEST_VERSION = "merlin_artist_term_idf_v1"
TAG_IDF_FORMULA = "log((artist_count + 1) / (artist_frequency + 1)) + 1"


@dataclass(frozen=True, slots=True)
class TagData:
    """Compact song ownership and bidirectional artist-tag mappings."""

    track_to_artist: Mapping[str, str]
    artist_tracks: Mapping[str, Sequence[str]]
    artist_terms: Mapping[str, frozenset[str]]
    term_artists: Mapping[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class SparseArtistTagIndex:
    """Exact normalized TF-IDF matrices for batched artist cosine queries."""

    artists: tuple[str, ...]
    artist_to_row: Mapping[str, int]
    normalized: object
    recall_queries: object

    @classmethod
    def build(
        cls,
        data: TagData,
        idf_values: Mapping[str, float],
        norms: Mapping[str, float],
        *,
        max_term_artists: int,
    ) -> SparseArtistTagIndex:
        from scipy.sparse import coo_matrix

        artists = tuple(sorted(data.artist_terms))
        terms = tuple(sorted(data.term_artists))
        artist_to_row = {artist: row for row, artist in enumerate(artists)}
        term_to_column = {term: column for column, term in enumerate(terms)}
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        recall_values: list[float] = []
        for artist in artists:
            norm = float(norms.get(artist, 0.0))
            if norm <= 0.0:
                continue
            row = artist_to_row[artist]
            for term in data.artist_terms[artist]:
                value = float(idf_values[term]) / norm
                rows.append(row)
                columns.append(term_to_column[term])
                values.append(value)
                recall_values.append(
                    value
                    if len(data.term_artists.get(term, ())) <= max_term_artists
                    else 0.0
                )
        shape = (len(artists), len(terms))
        normalized = coo_matrix(
            (values, (rows, columns)), shape=shape, dtype="float64"
        ).tocsr()
        recall_queries = coo_matrix(
            (recall_values, (rows, columns)), shape=shape, dtype="float64"
        ).tocsr()
        recall_queries.eliminate_zeros()
        return cls(artists, artist_to_row, normalized, recall_queries)

    def similar_many(
        self,
        artist_ids: Sequence[str],
        top_k: int,
        *,
        chunk_size: int = 64,
    ) -> dict[str, tuple[tuple[str, float], ...]]:
        """Return exact nonzero top artists for many roots in sparse chunks."""
        import numpy as np

        results = {artist_id: () for artist_id in artist_ids}
        available = [artist_id for artist_id in artist_ids if artist_id in self.artist_to_row]
        for start in range(0, len(available), chunk_size):
            chunk = available[start : start + chunk_size]
            rows = [self.artist_to_row[artist_id] for artist_id in chunk]
            similarities = self.recall_queries[rows] @ self.normalized.T
            for artist_id, result in zip(chunk, similarities, strict=True):
                valid = (
                    (result.data > 0.0)
                    & (result.indices != self.artist_to_row[artist_id])
                )
                candidate_rows = result.indices[valid]
                candidate_scores = result.data[valid]
                order = np.lexsort((candidate_rows, -candidate_scores))[:top_k]
                results[artist_id] = tuple(
                    (
                        self.artists[int(candidate_rows[position])],
                        float(candidate_scores[position]),
                    )
                    for position in order
                )
        return results

    def similarities(
        self,
        source_artist: str,
        target_artists: Sequence[str],
    ) -> list[float]:
        source = self.artist_to_row.get(source_artist)
        if source is None:
            return [0.0] * len(target_artists)
        positions = [
            index for index, artist in enumerate(target_artists)
            if artist in self.artist_to_row
        ]
        results = [0.0] * len(target_artists)
        if positions:
            rows = [self.artist_to_row[target_artists[index]] for index in positions]
            values = (self.normalized[source] @ self.normalized[rows].T).toarray()[0]
            for position, value in zip(positions, values, strict=True):
                results[position] = float(value)
        return results

    def similarities_many(
        self,
        targets_by_source: Mapping[str, Sequence[str]],
        *,
        chunk_size: int = 64,
    ) -> dict[str, list[float]]:
        """Score several source/target groups with one sparse product per chunk."""
        results = {
            source: [0.0] * len(targets)
            for source, targets in targets_by_source.items()
        }
        available = [
            source for source in targets_by_source if source in self.artist_to_row
        ]
        for start in range(0, len(available), chunk_size):
            sources = available[start : start + chunk_size]
            target_artists = tuple(dict.fromkeys(
                target
                for source in sources
                for target in targets_by_source[source]
                if target in self.artist_to_row
            ))
            if not target_artists:
                continue
            target_positions = {
                artist: position for position, artist in enumerate(target_artists)
            }
            source_rows = [self.artist_to_row[source] for source in sources]
            target_rows = [self.artist_to_row[target] for target in target_artists]
            similarities = (
                self.normalized[source_rows] @ self.normalized[target_rows].T
            )


def artist_tag_cosine(
    data: TagData,
    left_artist: str,
    right_artist: str,
    idf_values: Mapping[str, float] | None = None,
    norms: Mapping[str, float] | None = None,
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
    left_norm = (
        float(norms[left_artist])
        if norms is not None and left_artist in norms
        else math.sqrt(sum(idf(term) ** 2 for term in left))
    )
    right_norm = (
        float(norms[right_artist])
        if norms is not None and right_artist in norms
        else math.sqrt(sum(idf(term) ** 2 for term in right))
    )
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


def load_artist_term_data(
    graph_edges_path: str | Path,
    *,
    parquet_engine: str = "auto",
) -> TagData:
    """Load only canonical artist-term edges for the frozen IDF artifact."""
    rows = parquet_rows(
        graph_edges_path,
        ("src_id", "dst_id"),
        edge_type="artist_term",
        engine=parquet_engine,
    )
    return build_tag_data((), rows)


def _graph_edge_rows(path: str | Path, edge_type: str) -> Iterable[tuple[str, str]]:
    yield from parquet_rows(path, ("src_id", "dst_id"), edge_type=edge_type)


def _parquet_rows(
    path: str | Path,
    columns: tuple[str, str],
) -> Iterable[tuple[str, str]]:
    yield from parquet_rows(path, columns)


def find_similar_artists(
    data: TagData,
    artist_id: str,
    top_k: int,
    *,
    max_term_artists: int = 5_000,
    idf_values: Mapping[str, float] | None = None,
    norms: Mapping[str, float] | None = None,
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
    query_norm = (
        float(norms[artist_id])
        if norms is not None and artist_id in norms
        else math.sqrt(sum(idf(term) ** 2 for term in query_terms))
    )
    numerators: dict[str, float] = defaultdict(float)
    for term in usable_terms:
        weight = idf(term) ** 2
        for candidate in data.term_artists[term]:
            if candidate != artist_id:
                numerators[candidate] += weight

    scored = []
    for candidate, numerator in numerators.items():
        candidate_norm = (
            float(norms[candidate])
            if norms is not None and candidate in norms
            else math.sqrt(sum(idf(term) ** 2 for term in data.artist_terms[candidate]))
        )
        if candidate_norm > 0.0:
            scored.append((candidate, numerator / (query_norm * candidate_norm)))
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


def compute_artist_tag_norms(
    data: TagData,
    idf_values: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Precompute exact TF-IDF norms shared by recall and pair scoring."""
    artist_count = len(data.artist_terms)

    def idf(term: str) -> float:
        if idf_values is not None:
            try:
                return float(idf_values[term])
            except KeyError as error:
                raise ValueError(f"tag IDF artifact is missing term {term!r}") from error
        frequency = len(data.term_artists.get(term, ()))
        return math.log((1.0 + artist_count) / (1.0 + frequency)) + 1.0

    return {
        artist: math.sqrt(sum(idf(term) ** 2 for term in terms))
        for artist, terms in data.artist_terms.items()
    }


def compute_tag_idf(data: TagData) -> dict[str, float]:
    """Compute the canonical binary artist-term IDF table."""
    artist_count = len(data.artist_terms)
    if artist_count == 0 or not data.term_artists:
        raise ValueError("artist-term data must not be empty")
    values = {
        term: math.log((artist_count + 1.0) / (len(artists) + 1.0)) + 1.0
        for term, artists in sorted(data.term_artists.items())
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("computed tag IDF values must be finite and positive")
    return values


def build_tag_idf_artifact(
    data: TagData,
    graph_edges_path: str | Path,
) -> dict[str, object]:
    """Build a lineage-bound canonical Tag-IDF artifact."""
    graph_root = Path(graph_edges_path)
    artist_term_path = graph_root / "edge_type=artist_term"
    values = compute_tag_idf(data)
    return {
        "artifact_type": TAG_IDF_ARTIFACT_TYPE,
        "manifest_version": TAG_IDF_MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formula": TAG_IDF_FORMULA,
        "artist_count": len(data.artist_terms),
        "term_count": len(values),
        "source": {
            "artist_term_path": str(artist_term_path.resolve()),
            "artist_term_sha256": sha256_path(artist_term_path),
        },
        "values": values,
    }


def write_tag_idf_artifact(artifact: Mapping[str, object], path: str | Path) -> None:
    """Atomically publish a completed Tag-IDF artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(artifact), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)


def load_tag_idf(
    path: str | Path,
    *,
    expected_graph_edges_path: str | Path | None = None,
) -> dict[str, float]:
    """Load and validate the frozen artist-term IDF artifact."""
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
