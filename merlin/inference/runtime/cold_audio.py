from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..faiss_index import FaissTrackIndex
from ..types import Recommendation


@dataclass(frozen=True, slots=True)
class ColdAudioAudit:
    raw_results: int
    candidate_count: int
    candidate_shortage: int


@dataclass(slots=True)
class ColdAudioPipeline:
    audio_index: FaissTrackIndex
    track_to_song: Mapping[str, str] = field(default_factory=dict)
    candidate_limit: int = 1_000
    final_limit: int = 20
    overfetch_factor: int = 3

    def __post_init__(self) -> None:
        if self.candidate_limit != 1_000 or self.final_limit != 20:
            raise ValueError("cold Audio policy requires candidate_limit=1000 and final_limit=20")
        if self.overfetch_factor != 3:
            raise ValueError("cold Audio policy requires overfetch_factor=3")
        self.track_to_song = MappingProxyType(dict(self.track_to_song))

    def recommend(
        self,
        query_embedding: Any,
        k: int = 20,
        *,
        query_track_id: str | None = None,
        query_song_id: str | None = None,
    ) -> list[Recommendation]:
        recommendations, _audit = self.recommend_with_audit(
            query_embedding,
            k,
            query_track_id=query_track_id,
            query_song_id=query_song_id,
        )
        return recommendations

    def recommend_with_audit(
        self,
        query_embedding: Any,
        k: int = 20,
        *,
        query_track_id: str | None = None,
        query_song_id: str | None = None,
    ) -> tuple[list[Recommendation], ColdAudioAudit]:
        if not 1 <= k <= self.final_limit:
            raise ValueError("k must be between 1 and 20")
        requested = self.overfetch_factor * self.candidate_limit + 1
        raw = self.audio_index.search_vector(query_embedding, requested)
        filtered = []
        for track_id, score in raw:
            if track_id == query_track_id:
                continue
            if query_song_id is not None and self.track_to_song.get(track_id) == query_song_id:
                continue
            filtered.append((track_id, score))
            if len(filtered) == self.candidate_limit:
                break
        recommendations = [
            Recommendation(track_id, score, rank, frozenset({"audio"}), {"cos_audio": score})
            for rank, (track_id, score) in enumerate(filtered[:k], start=1)
        ]
        audit = ColdAudioAudit(len(raw), len(filtered), self.candidate_limit - len(filtered))
        return recommendations, audit
