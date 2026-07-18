"""Stage-1 candidate merging and lightweight retriever adapters."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .interfaces import CandidateRetriever
from .types import Candidate


def _different_song(_left: str, _right: str) -> bool:
    return False


def _zero_similarity(_left: str, _right: str) -> float:
    return 0.0


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
    overfetch_factor: int = 3

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query_track_id: str, limit: int) -> Sequence[Candidate]:
        if limit <= 0 or self.overfetch_factor <= 0:
            raise ValueError("vector recall limits must be positive")
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


@dataclass(slots=True)
class BfsRetriever(CandidateRetriever):
    """Artist-graph BFS with per-artist and total song caps."""

    track_to_artist: Mapping[str, str]
    artist_neighbors: Mapping[str, Sequence[str]]
    artist_tracks: Mapping[str, Sequence[str]]
    same_song: Callable[[str, str], bool] = _different_song
    tag_similarity: Callable[[str, str], float] = _zero_similarity
    max_depth: int = 2
    per_artist_cap: int = 10
    _name: str = "bfs"

    @classmethod
    def from_parquet(
        cls,
        songs_metadata_path: str,
        artist_edges_path: str,
        *,
        same_song: Callable[[str, str], bool] = _different_song,
        tag_similarity: Callable[[str, str], float] = _zero_similarity,
        max_depth: int = 2,
        per_artist_cap: int = 10,
    ) -> BfsRetriever:
        """Construct a retriever from the prepared runtime datasets."""
        from .bfs_data import load_bfs_data

        data = load_bfs_data(songs_metadata_path, artist_edges_path)
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

    def retrieve(self, query_track_id: str, limit: int) -> Sequence[Candidate]:
        root = self.track_to_artist.get(query_track_id)
        if root is None:
            return []
        queue = deque([(root, 0)])
        distances = {root: 0}
        while queue:
            artist, distance = queue.popleft()
            if distance < self.max_depth:
                for neighbor in self.artist_neighbors.get(artist, ()):
                    if neighbor not in distances:
                        distances[neighbor] = distance + 1
                        queue.append((neighbor, distance + 1))

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
    per_artist_cap: int = 5
    _name: str = "tag"

    @classmethod
    def from_parquet(
        cls,
        songs_metadata_path: str,
        artist_terms_path: str,
        *,
        tag_idf_path: str | None = None,
        same_song: Callable[[str, str], bool] = _different_song,
        artist_neighbor_limit: int = 100,
        max_term_artists: int = 5_000,
        per_artist_cap: int = 5,
    ) -> TagRetriever:
        """Construct lazy TF-IDF shared-tag recall from prepared datasets."""
        from .tag_data import find_similar_artists, load_tag_data, load_tag_idf

        data = load_tag_data(songs_metadata_path, artist_terms_path)
        idf_values = load_tag_idf(tag_idf_path) if tag_idf_path else None

        def neighbors(artist_id: str) -> Sequence[tuple[str, float]]:
            return find_similar_artists(
                data,
                artist_id,
                artist_neighbor_limit,
                max_term_artists=max_term_artists,
                idf_values=idf_values,
            )

        return cls(
            track_to_artist=data.track_to_artist,
            similar_artists=neighbors,
            artist_tracks=data.artist_tracks,
            same_song=same_song,
            per_artist_cap=per_artist_cap,
        )

    @property
    def name(self) -> str:
        return self._name

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
