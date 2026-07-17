"""Stage-1 candidate merging and lightweight retriever adapters."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .interfaces import CandidateRetriever
from .types import Candidate


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

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query_track_id: str, limit: int) -> Sequence[Candidate]:
        return [
            Candidate(
                track_id=track_id,
                sources=frozenset({self.name}),
                recall_scores={self.name: float(score)},
                source_ranks={self.name: rank},
            )
            for rank, (track_id, score) in enumerate(
                self.search(query_track_id, limit + 1), start=1
            )
            if track_id != query_track_id
        ][:limit]


@dataclass(slots=True)
class BfsRetriever(CandidateRetriever):
    """Artist-graph BFS with per-artist and total song caps."""

    track_to_artist: Mapping[str, str]
    artist_neighbors: Mapping[str, Sequence[str]]
    artist_tracks: Mapping[str, Sequence[str]]
    max_depth: int = 2
    per_artist_cap: int = 10
    _name: str = "bfs"

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query_track_id: str, limit: int) -> Sequence[Candidate]:
        root = self.track_to_artist.get(query_track_id)
        if root is None:
            return []
        queue = deque([(root, 0)])
        visited = {root}
        result: list[Candidate] = []
        while queue and len(result) < limit:
            artist, distance = queue.popleft()
            if distance > 0:
                score = 1.0 / (1.0 + distance)
                for track_id in self.artist_tracks.get(artist, ())[: self.per_artist_cap]:
                    if track_id != query_track_id:
                        result.append(Candidate(
                            track_id=track_id,
                            sources=frozenset({self.name}),
                            recall_scores={self.name: score},
                            source_ranks={self.name: len(result) + 1},
                        ))
                        if len(result) == limit:
                            break
            if distance < self.max_depth:
                for neighbor in self.artist_neighbors.get(artist, ()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
        return result


@dataclass(slots=True)
class TagRetriever(CandidateRetriever):
    """Adapter for a precomputed TF-IDF artist-neighbor table."""

    track_to_artist: Mapping[str, str]
    similar_artists: Mapping[str, Sequence[tuple[str, float]]]
    artist_tracks: Mapping[str, Sequence[str]]
    per_artist_cap: int = 5
    _name: str = "tag"

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query_track_id: str, limit: int) -> Sequence[Candidate]:
        root = self.track_to_artist.get(query_track_id)
        if root is None:
            return []
        result: list[Candidate] = []
        for artist, similarity in self.similar_artists.get(root, ()):
            for track_id in self.artist_tracks.get(artist, ())[: self.per_artist_cap]:
                if track_id == query_track_id:
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
