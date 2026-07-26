"""Canonical Ranker feature contract, computation, and metadata loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ...artifacts.io import parquet_rows
from ...types import Candidate


FEATURE_SCHEMA = "ranker-v2"
FEATURE_ORDER = (
    "cos_audio",
    "cos_graph",
    "has_graph",
    "bfs_score",
    "has_bfs",
    "tag_tfidf_cosine",
    "has_tags",
    "same_release",
    "has_release",
    "year_gap",
    "has_year",
    "audio_tag_interaction",
    "graph_bfs_interaction",
)
RAW_FEATURE_ORDER = FEATURE_ORDER[:11]


PairLookup = Callable[[str, str], float | None]
BatchPairLookup = Callable[[str, Sequence[str]], Sequence[float | None]]
PairListLookup = Callable[
    [Sequence[tuple[str, str]]],
    Sequence[float | None],
]


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    """Metadata whose availability is meaningful to the Ranker."""

    release_id: str | None = None
    year: int | None = None


def build_track_metadata(
    rows: Iterable[tuple[str, object, int | None, bool]],
) -> dict[str, TrackMetadata]:
    """Build metadata from real release IDs and the prepared year mask."""
    tracks: dict[str, TrackMetadata] = {}
    for track_id, release_id, year, has_year in rows:
        if not track_id:
            continue
        release = str(release_id) if release_id not in (None, "", 0, "0") else None
        metadata = TrackMetadata(
            release_id=release,
            year=int(year) if has_year and year is not None else None,
        )
        previous = tracks.setdefault(track_id, metadata)
        if previous != metadata:
            raise ValueError(f"track {track_id!r} has conflicting ranker metadata")
    return tracks


def load_track_metadata(path: str | Path) -> dict[str, TrackMetadata]:
    """Read only Ranker metadata columns in bounded Parquet batches."""
    columns = ["track_id", "release_7digitalid", "year", "has_year"]
    return build_track_metadata(parquet_rows(path, columns))


@dataclass(frozen=True, slots=True)
class PairSignalLookups:
    """Compute pair signals independently of candidate provenance."""

    audio: PairLookup
    graph: PairLookup
    bfs: PairLookup
    tags: PairLookup
    audio_batch: BatchPairLookup | None = None
    graph_batch: BatchPairLookup | None = None
    bfs_batch: BatchPairLookup | None = None
    tags_batch: BatchPairLookup | None = None
    audio_pairs: PairListLookup | None = None
    graph_pairs: PairListLookup | None = None
    bfs_pairs: PairListLookup | None = None
    tags_pairs: PairListLookup | None = None


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
        if artifact.get("feature_schema_version") != FEATURE_SCHEMA:
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
class RankerFeatureComputer:
    """Compute features for every candidate in the canonical union."""

    tracks: Mapping[str, TrackMetadata]
    signals: PairSignalLookups
    fills: FeatureFillValues = field(default_factory=FeatureFillValues)
    schema_version: str = FEATURE_SCHEMA

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
        audio_values = self._signal_values(
            query_track_id,
            candidates,
            "audio",
            self.signals.audio,
            self.signals.audio_batch,
        )
        graph_values = self._signal_values(
            query_track_id,
            candidates,
            "graph",
            self.signals.graph,
            self.signals.graph_batch,
        )
        bfs_values = self._signal_values(
            query_track_id,
            candidates,
            "bfs",
            self.signals.bfs,
            self.signals.bfs_batch,
        )
        tag_values = self._signal_values(
            query_track_id,
            candidates,
            "tag",
            self.signals.tags,
            self.signals.tags_batch,
        )
        sizes = {len(audio_values), len(graph_values), len(bfs_values), len(tag_values)}
        if sizes != {len(candidates)}:
            raise ValueError("batch pair lookup returned the wrong number of scores")
        return [
            self._raw_values(
                query_track_id,
                candidate_id,
                _finite(audio),
                _finite(graph),
                _finite(bfs),
                _finite(tags),
            )
            for candidate_id, audio, graph, bfs, tags in zip(
                candidate_ids,
                audio_values,
                graph_values,
                bfs_values,
                tag_values,
                strict=True,
            )
        ]

    def compute_raw_ids(
        self,
        query_track_id: str,
        candidate_ids: Sequence[str],
    ) -> list[Mapping[str, float | None]]:
        """Compute a candidate-ID batch without allocating Candidate wrappers."""
        values = [
            self._id_signal_values(query_track_id, candidate_ids, lookup, batch_lookup)
            for lookup, batch_lookup in (
                (self.signals.audio, self.signals.audio_batch),
                (self.signals.graph, self.signals.graph_batch),
                (self.signals.bfs, self.signals.bfs_batch),
                (self.signals.tags, self.signals.tags_batch),
            )
        ]
        return [
            self._raw_values(query_track_id, candidate_id, *signals)
            for candidate_id, signals in zip(
                candidate_ids,
                zip(*values, strict=True),
                strict=True,
            )
        ]

    def compute_raw_pairs(
        self,
        pairs: Sequence[tuple[str, str, Mapping[str, float]]],
    ) -> list[Mapping[str, float | None]]:
        """Compute arbitrary query/candidate pairs in one bounded signal batch."""
        columns = self.compute_raw_pair_columns(pairs)
        return [
            dict(zip(RAW_FEATURE_ORDER, values, strict=True))
            for values in zip(
                *(columns[name] for name in RAW_FEATURE_ORDER),
                strict=True,
            )
        ]

    def compute_raw_pair_columns(
        self,
        pairs: Sequence[tuple[str, str, Mapping[str, float]]],
    ) -> dict[str, list[float | None]]:
        """Compute arbitrary pair features without allocating one dict per row."""
        values = [
            self._pair_values(pairs, source, lookup, pair_lookup)
            for source, lookup, pair_lookup in (
                ("audio", self.signals.audio, self.signals.audio_pairs),
                ("graph", self.signals.graph, self.signals.graph_pairs),
                ("bfs", self.signals.bfs, self.signals.bfs_pairs),
                ("tag", self.signals.tags, self.signals.tags_pairs),
            )
        ]
        columns: dict[str, list[float | None]] = {
            name: [] for name in RAW_FEATURE_ORDER
        }
        for (query_id, candidate_id, _hints), signals in zip(
            pairs,
            zip(*values, strict=True),
            strict=True,
        ):
            for name, value in zip(
                RAW_FEATURE_ORDER,
                self._raw_tuple(query_id, candidate_id, *signals),
                strict=True,
            ):
                columns[name].append(value)
        return columns

    def _raw_tuple(
        self,
        query_track_id: str,
        candidate_id: str,
        audio_raw: float | None,
        graph_raw: float | None,
        bfs_raw: float | None,
        tags_raw: float | None,
    ) -> tuple[float | None, ...]:
        query = self.tracks.get(query_track_id, TrackMetadata())
        other = self.tracks.get(candidate_id, TrackMetadata())
        has_release = query.release_id is not None and other.release_id is not None
        has_year = query.year is not None and other.year is not None
        year_gap_raw = float(abs(query.year - other.year)) if has_year else None
        return (
            audio_raw,
            graph_raw,
            float(graph_raw is not None),
            bfs_raw,
            float(bfs_raw is not None),
            tags_raw,
            float(tags_raw is not None),
            float(has_release and query.release_id == other.release_id),
            float(has_release),
            year_gap_raw,
            float(has_year),
        )

    @staticmethod
    def _signal_values(
        query_track_id: str,
        candidates: Sequence[Candidate],
        source: str,
        lookup: PairLookup,
        batch_lookup: BatchPairLookup | None,
    ) -> list[float | None]:
        """Reuse exact recall scores and batch-compute only missing pair signals."""
        values: list[float | None] = [None] * len(candidates)
        missing_indexes: list[int] = []
        missing_ids: list[str] = []
        for index, candidate in enumerate(candidates):
            hinted = candidate.recall_scores.get(source)
            if hinted is None:
                missing_indexes.append(index)
                missing_ids.append(candidate.track_id)
            else:
                values[index] = _finite(hinted)
        if missing_ids:
            computed = (
                batch_lookup(query_track_id, missing_ids)
                if batch_lookup is not None
                else [lookup(query_track_id, candidate_id) for candidate_id in missing_ids]
            )
            if len(computed) != len(missing_ids):
                raise ValueError("batch pair lookup returned the wrong number of scores")
            for index, value in zip(missing_indexes, computed, strict=True):
                values[index] = _finite(value)
        return values

    @staticmethod
    def _id_signal_values(
        query_track_id: str,
        candidate_ids: Sequence[str],
        lookup: PairLookup,
        batch_lookup: BatchPairLookup | None,
    ) -> list[float | None]:
        computed = (
            batch_lookup(query_track_id, candidate_ids)
            if batch_lookup is not None
            else [lookup(query_track_id, candidate_id) for candidate_id in candidate_ids]
        )
        if len(computed) != len(candidate_ids):
            raise ValueError("batch pair lookup returned the wrong number of scores")
        return [_finite(value) for value in computed]

    @staticmethod
    def _pair_values(
        pairs: Sequence[tuple[str, str, Mapping[str, float]]],
        source: str,
        lookup: PairLookup,
        pair_lookup: PairListLookup | None,
    ) -> list[float | None]:
        values: list[float | None] = [None] * len(pairs)
        missing_indexes: list[int] = []
        missing_pairs: list[tuple[str, str]] = []
        for index, (query_id, candidate_id, hints) in enumerate(pairs):
            hinted = hints.get(source)
            if hinted is None:
                missing_indexes.append(index)
                missing_pairs.append((query_id, candidate_id))
            else:
                values[index] = _finite(hinted)
        if missing_pairs:
            computed = (
                pair_lookup(missing_pairs)
                if pair_lookup is not None
                else [lookup(query_id, candidate_id) for query_id, candidate_id in missing_pairs]
            )
            if len(computed) != len(missing_pairs):
                raise ValueError("pair-list lookup returned the wrong number of scores")
            for index, value in zip(missing_indexes, computed, strict=True):
                values[index] = _finite(value)
        return values

    def _raw_values(
        self,
        query_track_id: str,
        candidate_id: str,
        audio_raw: float | None,
        graph_raw: float | None,
        bfs_raw: float | None,
        tags_raw: float | None,
    ) -> Mapping[str, float | None]:
        return dict(zip(
            RAW_FEATURE_ORDER,
            self._raw_tuple(
                query_track_id,
                candidate_id,
                audio_raw,
                graph_raw,
                bfs_raw,
                tags_raw,
            ),
            strict=True,
        ))

    def _fill(self, name: str, value: float | None) -> float:
        return self.fills.get(name) if value is None else value


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None
