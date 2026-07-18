"""Ranker-v1 query-candidate feature computation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from .types import Candidate


RANKER_V1_FEATURES = (
    "cos_audio",
    "cos_graph",
    "bfs_score",
    "tag_tfidf_cosine",
    "same_album",
    "same_year",
    "year_gap",
    "candidate_popularity",
    "from_audio",
    "from_graph",
    "from_bfs",
    "from_tag",
)


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    album_key: str | None = None
    year: int | None = None
    popularity: float | None = None


def build_track_metadata(
    rows: Iterable[tuple[str, str | None, int | None, bool, float | None]],
) -> dict[str, TrackMetadata]:
    """Build ranker metadata while applying the prepared ``has_year`` mask."""
    tracks: dict[str, TrackMetadata] = {}
    for track_id, album_key, year, has_year, popularity in rows:
        if not track_id:
            continue
        metadata = TrackMetadata(
            album_key=str(album_key) if album_key else None,
            year=int(year) if has_year and year is not None else None,
            popularity=_finite_or_zero(popularity),
        )
        previous = tracks.setdefault(track_id, metadata)
        if previous != metadata:
            raise ValueError(f"track {track_id!r} has conflicting metadata")
    return tracks


@dataclass(frozen=True, slots=True)
class InferenceFeatureComputer:
    """Compute the exact named feature contract shared with Person A."""

    tracks: Mapping[str, TrackMetadata]
    schema_version: str = "ranker-v1"

    def compute(self, query_track_id: str, candidate: Candidate) -> Mapping[str, float]:
        query = self.tracks.get(query_track_id, TrackMetadata())
        other = self.tracks.get(candidate.track_id, TrackMetadata())
        same_album = bool(query.album_key and query.album_key == other.album_key)
        have_years = query.year is not None and other.year is not None
        same_year = have_years and query.year == other.year
        year_gap = abs(query.year - other.year) if have_years else 0
        scores = candidate.recall_scores
        sources = candidate.sources
        return {
            "cos_audio": _finite_or_zero(scores.get("audio")),
            "cos_graph": _finite_or_zero(scores.get("graph")),
            "bfs_score": _finite_or_zero(scores.get("bfs")),
            "tag_tfidf_cosine": _finite_or_zero(scores.get("tag")),
            "same_album": float(same_album),
            "same_year": float(same_year),
            "year_gap": float(year_gap),
            "candidate_popularity": _finite_or_zero(other.popularity),
            "from_audio": float("audio" in sources),
            "from_graph": float("graph" in sources),
            "from_bfs": float("bfs" in sources),
            "from_tag": float("tag" in sources),
        }


def _finite_or_zero(value: float | None) -> float:
    number = 0.0 if value is None else float(value)
    return number if math.isfinite(number) else 0.0
