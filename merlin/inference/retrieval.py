"""Stage-1 candidate merging and lightweight retriever adapters."""

from __future__ import annotations

from collections import deque, OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Mapping, Sequence

from .interfaces import CandidateRetriever
from .types import Candidate


def _different_song(_left: str, _right: str) -> bool:
    return False


def _zero_similarity(_left: str, _right: str) -> float:
    return 0.0


def _available(_track_id: str) -> bool:
    return True


def merge_candidates(groups: Sequence[Sequence[Candidate]], query_track_id: str) -> list[Candidate]:
    """Union candidates by track ID while preserving all recall evidence."""
    merged: dict[str, dict[str, object]] = {}
    for group in groups:
        for candidate in group:
            if candidate.track_id == query_track_id:
                continue
            state = merged.setdefault(
                candidate.track_id,
                {"sources": set(), "scores": {}, "ranks": {}},
            )
            state["sources"].update(candidate.sources)  # type: ignore[union-attr]
            state["scores"].update(candidate.recall_scores)  # type: ignore[union-attr]
            state["ranks"].update(candidate.source_ranks)  # type: ignore[union-attr]
    return [
        Candidate(
            track_id=track_id,
            sources=frozenset(state["sources"]),  # type: ignore[arg-type]
            recall_scores=state["scores"],  # type: ignore[arg-type]
            source_ranks=state["ranks"],  # type: ignore[arg-type]
        )
        for track_id, state in merged.items()
    ]


@dataclass(slots=True)
class VectorRetriever(CandidateRetriever):
    """Adapter around an Audio/Graph nearest-neighbor search function.

    ``search`` owns index-specific details and returns ``(track_id, score)``
    ordered from best to worst. This keeps FAISS optional at package import time.
    """

    _name: str
    search: Callable[[str, int], Sequence[tuple[str, float]]]
    same_song: Callable[[str, str], bool] = _different_song
    query_available: Callable[[str], bool] = _available
    overfetch_factor: int = 3

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query_track_id: str, limit: int) -> Sequence[Candidate]:
        if limit <= 0 or self.overfetch_factor <= 0:
            raise ValueError("vector recall limits must be positive")
        if not self.is_available(query_track_id):
            return []
        result: list[Candidate] = []
        seen: set[str] = set()
        neighbors = self.search(query_track_id, self.overfetch_factor * limit + 1)
        for track_id, score in neighbors:
            if (
                track_id == query_track_id
                or track_id in seen
                or self.same_song(query_track_id, track_id)
            ):
                continue
            seen.add(track_id)
            result.append(Candidate(
                track_id=track_id,
                sources=frozenset({self.name}),
                recall_scores={self.name: float(score)},
                source_ranks={self.name: len(result) + 1},
            ))
            if len(result) == limit:
                break
        return result

    def is_available(self, query_track_id: str) -> bool:
        return bool(self.query_available(query_track_id))


@dataclass(slots=True)
class BfsRetriever(CandidateRetriever):
    """Artist-graph BFS with per-artist and total song caps."""

    track_to_artist: Mapping[str, str]
    artist_neighbors: Mapping[str, Sequence[str]]
    artist_tracks: Mapping[str, Sequence[str]]
    same_song: Callable[[str, str], bool] = _different_song
    pair_similarity: Callable[[str, str], float | None] = _zero_similarity
    tag_similarity: Callable[[str, str], float] = _zero_similarity
    max_depth: int = 2
    per_artist_cap: int = 10
    _name: str = "bfs"
    _distance_cache: OrderedDict[str, dict[str, int]] = field(
        init=False,
        repr=False,
        default_factory=OrderedDict,
    )

    @classmethod
    def from_parquet(
        cls,
        songs_metadata_path: str,
        graph_edges_path: str,
        *,
        same_song: Callable[[str, str], bool] = _different_song,
        tag_similarity: Callable[[str, str], float] = _zero_similarity,
        max_depth: int = 2,
        per_artist_cap: int = 10,
    ) -> BfsRetriever:
        """Construct a retriever from the prepared runtime datasets."""
        from .bfs_data import load_bfs_data

        data = load_bfs_data(songs_metadata_path, graph_edges_path)
        return cls(
            track_to_artist=data.track_to_artist,
            artist_neighbors=data.artist_neighbors,
            artist_tracks=data.artist_tracks,
            same_song=same_song,
            tag_similarity=tag_similarity,
            max_depth=max_depth,
            per_artist_cap=per_artist_cap,
        )

    @property
    def name(self) -> str:
        return self._name

    def pair_score(self, left_track_id: str, right_track_id: str) -> float | None:
        """Return the directed artist-BFS score independently of recall source."""
        source = self.track_to_artist.get(left_track_id)
        target = self.track_to_artist.get(right_track_id)
        if source is None or target is None or source == target:
            return None
        distance = self._distances(source).get(target)
        return None if distance is None else 1.0 / (1.0 + distance)

    def _distances(self, source: str) -> dict[str, int]:
        cached = self._distance_cache.get(source)
        if cached is not None:
            self._distance_cache.move_to_end(source)
            return cached
        queue = deque([(source, 0)])
        distances = {source: 0}
        while queue:
            artist, distance = queue.popleft()
            if distance == self.max_depth:
                continue
            next_distance = distance + 1
            for neighbor in self.artist_neighbors.get(artist, ()):
                if neighbor not in distances:
                    distances[neighbor] = next_distance
                    queue.append((neighbor, next_distance))
        self._distance_cache[source] = distances
        if len(self._distance_cache) > 4_096:
            self._distance_cache.popitem(last=False)
        return distances

    def is_available(self, query_track_id: str) -> bool:
        return query_track_id in self.track_to_artist

    def retrieve(self, query_track_id: str, limit: int) -> Sequence[Candidate]:
        root = self.track_to_artist.get(query_track_id)
        if root is None:
            return []
        distances = self._distances(root)

        ordered: list[tuple[int, float, str]] = []
        for artist, distance in distances.items():
            if distance == 0:
                continue
            similarity = float(self.tag_similarity(root, artist))
            tracks = sorted(
                track_id for track_id in self.artist_tracks.get(artist, ())
                if track_id != query_track_id and not self.same_song(query_track_id, track_id)
            )[: self.per_artist_cap]
            ordered.extend((distance, similarity, track_id) for track_id in tracks)
        ordered.sort(key=lambda item: (item[0], -item[1], item[2]))
        return [
            Candidate(
                track_id=track_id,
                sources=frozenset({self.name}),
                recall_scores={self.name: 1.0 / (1.0 + distance)},
                source_ranks={self.name: rank},
            )
            for rank, (distance, _similarity, track_id) in enumerate(
                ordered[:limit], start=1
            )
        ]


@dataclass(slots=True)
class TagRetriever(CandidateRetriever):
    """Adapter for a precomputed TF-IDF artist-neighbor table."""

    track_to_artist: Mapping[str, str]
    similar_artists: (
        Mapping[str, Sequence[tuple[str, float]]]
        | Callable[[str], Sequence[tuple[str, float]]]
    )
    artist_tracks: Mapping[str, Sequence[str]]
    same_song: Callable[[str, str], bool] = _different_song
    pair_similarity: Callable[[str, str], float | None] = _zero_similarity
    query_available: Callable[[str], bool] | None = None
    per_artist_cap: int = 5
    _name: str = "tag"

    @classmethod
    def from_parquet(
        cls,
        songs_metadata_path: str,
        graph_edges_path: str,
        *,
        tag_idf_path: str | None = None,
        same_song: Callable[[str, str], bool] = _different_song,
        artist_neighbor_limit: int = 100,
        max_term_artists: int = 5_000,
        per_artist_cap: int = 5,
    ) -> TagRetriever:
        """Construct lazy TF-IDF shared-tag recall from prepared datasets."""
        from .tag_data import load_tag_data, load_tag_idf

        data = load_tag_data(songs_metadata_path, graph_edges_path)
        idf_values = (
            load_tag_idf(
                tag_idf_path,
                expected_graph_edges_path=graph_edges_path,
            )
            if tag_idf_path
            else None
        )
        return cls.from_data(
            data,
            idf_values=idf_values,
            same_song=same_song,
            artist_neighbor_limit=artist_neighbor_limit,
            max_term_artists=max_term_artists,
            per_artist_cap=per_artist_cap,
        )

    @classmethod
    def from_data(
        cls,
        data,
        *,
        idf_values: Mapping[str, float] | None = None,
        same_song: Callable[[str, str], bool] = _different_song,
        artist_neighbor_limit: int = 100,
        max_term_artists: int = 5_000,
        per_artist_cap: int = 5,
    ) -> TagRetriever:
        """Construct a retriever from catalog data already loaded by a batch stage."""
        from .tag_data import artist_tag_cosine, compute_artist_tag_norms, find_similar_artists

        norms = compute_artist_tag_norms(data, idf_values)

        @lru_cache(maxsize=50_000)
        def neighbors(artist_id: str) -> Sequence[tuple[str, float]]:
            return tuple(find_similar_artists(
                data,
                artist_id,
                artist_neighbor_limit,
                max_term_artists=max_term_artists,
                idf_values=idf_values,
                norms=norms,
            ))

        @lru_cache(maxsize=500_000)
        def canonical_pair_similarity(left: str, right: str) -> float | None:
            return artist_tag_cosine(data, left, right, idf_values, norms)

        def pair_similarity(left_artist: str, right_artist: str) -> float | None:
            left, right = sorted((left_artist, right_artist))
            return canonical_pair_similarity(left, right)

        return cls(
            track_to_artist=data.track_to_artist,
            similar_artists=neighbors,
            artist_tracks=data.artist_tracks,
            same_song=same_song,
            pair_similarity=pair_similarity,
            query_available=lambda track_id: (
                (artist_id := data.track_to_artist.get(track_id)) is not None
                and artist_id in data.artist_terms
            ),
            per_artist_cap=per_artist_cap,
        )

    @property
    def name(self) -> str:
        return self._name

    def pair_score(self, left_track_id: str, right_track_id: str) -> float | None:
        left = self.track_to_artist.get(left_track_id)
        right = self.track_to_artist.get(right_track_id)
        if left is None or right is None:
            return None
        return self.pair_similarity(left, right)

    def artist_similarity(self, left_artist: str, right_artist: str) -> float:
        score = self.pair_similarity(left_artist, right_artist)
        return 0.0 if score is None else float(score)

    def is_available(self, query_track_id: str) -> bool:
        if self.query_available is not None:
            return bool(self.query_available(query_track_id))
        artist = self.track_to_artist.get(query_track_id)
        return artist is not None and bool(self._artist_terms_available(artist))

    def _artist_terms_available(self, artist_id: str) -> bool:
        if callable(self.similar_artists):
            return True
        return artist_id in self.similar_artists

    def retrieve(self, query_track_id: str, limit: int) -> Sequence[Candidate]:
        root = self.track_to_artist.get(query_track_id)
        if root is None:
            return []
        result: list[Candidate] = []
        neighbors = (
            self.similar_artists(root)
            if callable(self.similar_artists)
            else self.similar_artists.get(root, ())
        )
        for artist, similarity in neighbors:
            for track_id in self.artist_tracks.get(artist, ())[: self.per_artist_cap]:
                if track_id == query_track_id or self.same_song(query_track_id, track_id):
                    continue
                result.append(Candidate(
                    track_id=track_id,
                    sources=frozenset({self.name}),
                    recall_scores={self.name: float(similarity)},
                    source_ranks={self.name: len(result) + 1},
                ))
                if len(result) == limit:
                    return result
        return result
