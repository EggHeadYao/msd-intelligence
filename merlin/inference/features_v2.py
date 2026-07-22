"""Ranker-v2 pair feature computation and metadata loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .feature_schema import RANKER_V2_SCHEMA_VERSION
from .parquet_io import parquet_rows
from .types import Candidate


PairLookup = Callable[[str, str], float | None]
BatchPairLookup = Callable[[str, Sequence[str]], Sequence[float | None]]


@dataclass(frozen=True, slots=True)
class TrackMetadataV2:
    """Metadata whose availability is meaningful to the v2 ranker."""

    release_id: str | None = None
    year: int | None = None


def build_track_metadata_v2(
    rows: Iterable[tuple[str, object, int | None, bool]],
) -> dict[str, TrackMetadataV2]:
    """Build metadata from real release IDs and the prepared year mask."""
    tracks: dict[str, TrackMetadataV2] = {}
    for track_id, release_id, year, has_year in rows:
        if not track_id:
            continue
        release = str(release_id) if release_id not in (None, "", 0, "0") else None
        metadata = TrackMetadataV2(
            release_id=release,
            year=int(year) if has_year and year is not None else None,
        )
        previous = tracks.setdefault(track_id, metadata)
        if previous != metadata:
            raise ValueError(f"track {track_id!r} has conflicting v2 metadata")
    return tracks


def load_track_metadata_v2(path: str | Path) -> dict[str, TrackMetadataV2]:
    """Read only ranker-v2 metadata columns in bounded Parquet batches."""
    columns = ["track_id", "release_7digitalid", "year", "has_year"]
    return build_track_metadata_v2(parquet_rows(path, columns))


@dataclass(frozen=True, slots=True)
class PairSignalLookups:
    """Compute pair signals independently of candidate provenance."""

    audio: PairLookup
    graph: PairLookup
    bfs: PairLookup
    tags: PairLookup
    audio_batch: BatchPairLookup | None = None
    graph_batch: BatchPairLookup | None = None


@dataclass(frozen=True, slots=True)
class FeatureFillValues:
    """Set-A statistics used when a continuous pair signal is unavailable."""

    values: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        feature_order: tuple[str, ...],
    ) -> FeatureFillValues:
        with Path(path).open("r", encoding="utf-8") as stream:
            artifact = json.load(stream)
        if artifact.get("feature_schema_version") != RANKER_V2_SCHEMA_VERSION:
            raise ValueError("fill artifact schema version mismatch")
        if tuple(artifact.get("feature_order", ())) != feature_order:
            raise ValueError("fill artifact feature order mismatch")
        values = {
            str(name): float(value)
            for name, value in artifact.get("fill_values", {}).items()
        }
        if any(name not in feature_order for name in values):
            raise ValueError("fill artifact contains an unknown feature")
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("fill artifact contains a non-finite value")
        return cls(values)

    def get(self, feature_name: str) -> float:
        return float(self.values.get(feature_name, 0.0))


@dataclass(frozen=True, slots=True)
class RankerV2FeatureComputer:
    """Compute v2 features for every candidate in the canonical union."""

    tracks: Mapping[str, TrackMetadataV2]
    signals: PairSignalLookups
    fills: FeatureFillValues = field(default_factory=FeatureFillValues)
    schema_version: str = RANKER_V2_SCHEMA_VERSION

    def compute(self, query_track_id: str, candidate: Candidate) -> Mapping[str, float]:
        raw = self.compute_raw(query_track_id, candidate)
        audio = self._fill("cos_audio", raw["cos_audio"])
        graph = self._fill("cos_graph", raw["cos_graph"])
        bfs = self._fill("bfs_score", raw["bfs_score"])
        tags = self._fill("tag_tfidf_cosine", raw["tag_tfidf_cosine"])
        year_gap = self._fill("year_gap", raw["year_gap"])
        return {
            "cos_audio": audio,
            "cos_graph": graph,
            "has_graph": float(raw["has_graph"]),
            "bfs_score": bfs,
            "has_bfs": float(raw["has_bfs"]),
            "tag_tfidf_cosine": tags,
            "has_tags": float(raw["has_tags"]),
            "same_release": float(raw["same_release"]),
            "has_release": float(raw["has_release"]),
            "year_gap": year_gap,
            "has_year": float(raw["has_year"]),
            "audio_tag_interaction": audio * tags,
            "graph_bfs_interaction": graph * bfs,
        }

    def compute_raw(
        self,
        query_track_id: str,
        candidate: Candidate,
    ) -> Mapping[str, float | None]:
        """Return pre-fill base signals for Set-A statistics and training."""
        candidate_id = candidate.track_id
        audio_raw = _finite(self.signals.audio(query_track_id, candidate_id))
        graph_raw = _finite(self.signals.graph(query_track_id, candidate_id))
        bfs_raw = _finite(self.signals.bfs(query_track_id, candidate_id))
        tags_raw = _finite(self.signals.tags(query_track_id, candidate_id))
        return self._raw_values(
            query_track_id,
            candidate_id,
            audio_raw,
            graph_raw,
            bfs_raw,
            tags_raw,
        )

    def compute_raw_many(
        self,
        query_track_id: str,
        candidates: Sequence[Candidate],
    ) -> list[Mapping[str, float | None]]:
        """Compute one query's features with optional batched vector lookups."""
        candidate_ids = [candidate.track_id for candidate in candidates]
        audio_values = (
            self.signals.audio_batch(query_track_id, candidate_ids)
            if self.signals.audio_batch is not None
            else [self.signals.audio(query_track_id, candidate_id) for candidate_id in candidate_ids]
        )
        graph_values = (
            self.signals.graph_batch(query_track_id, candidate_ids)
            if self.signals.graph_batch is not None
            else [self.signals.graph(query_track_id, candidate_id) for candidate_id in candidate_ids]
        )
        if len(audio_values) != len(candidates) or len(graph_values) != len(candidates):
            raise ValueError("batch pair lookup returned the wrong number of scores")
        return [
            self._raw_values(
                query_track_id,
                candidate_id,
                _finite(audio),
                _finite(graph),
                _finite(self.signals.bfs(query_track_id, candidate_id)),
                _finite(self.signals.tags(query_track_id, candidate_id)),
            )
            for candidate_id, audio, graph in zip(
                candidate_ids,
                audio_values,
                graph_values,
                strict=True,
            )
        ]

    def _raw_values(
        self,
        query_track_id: str,
        candidate_id: str,
        audio_raw: float | None,
        graph_raw: float | None,
        bfs_raw: float | None,
        tags_raw: float | None,
    ) -> Mapping[str, float | None]:
        query = self.tracks.get(query_track_id, TrackMetadataV2())
        other = self.tracks.get(candidate_id, TrackMetadataV2())
        has_release = query.release_id is not None and other.release_id is not None
        has_year = query.year is not None and other.year is not None
        year_gap_raw = float(abs(query.year - other.year)) if has_year else None
        return {
            "cos_audio": audio_raw,
            "cos_graph": graph_raw,
            "has_graph": float(graph_raw is not None),
            "bfs_score": bfs_raw,
            "has_bfs": float(bfs_raw is not None),
            "tag_tfidf_cosine": tags_raw,
            "has_tags": float(tags_raw is not None),
            "same_release": float(has_release and query.release_id == other.release_id),
            "has_release": float(has_release),
            "year_gap": year_gap_raw,
            "has_year": float(has_year),
        }

    def _fill(self, name: str, value: float | None) -> float:
        return self.fills.get(name) if value is None else value


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None
