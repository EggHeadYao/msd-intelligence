"""Encoded, batched recall used by high-volume offline stages."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..data.catalog import SameSongFilter
from ..retrieval.faiss import FaissTrackIndex
from ..retrieval import BfsRetriever, TagRetriever
from ..types import RecallAudit


SOURCE_NAMES = ("audio", "graph", "bfs", "tag")


@dataclass(frozen=True, slots=True)
class TrackCodec:
    """Stable integer IDs and vectorized catalog membership metadata."""

    tracks: tuple[str, ...]
    track_to_code: Mapping[str, int]
    allowed: np.ndarray
    song_codes: np.ndarray

    @classmethod
    def build(
        cls,
        assignments: Mapping[str, str],
        allowed_splits: frozenset[str],
        same_song: SameSongFilter,
    ) -> TrackCodec:
        tracks = tuple(sorted(assignments))
        track_to_code = {track_id: code for code, track_id in enumerate(tracks)}
        allowed = np.fromiter(
            (assignments[track_id] in allowed_splits for track_id in tracks),
            dtype=np.bool_,
            count=len(tracks),
        )
        song_to_code: dict[str, int] = {}
        song_codes = np.full(len(tracks), -1, dtype=np.int32)
        for track_id, song_id in same_song.track_to_song.items():
            track_code = track_to_code.get(track_id)
            if track_code is None:
                continue
            song_codes[track_code] = song_to_code.setdefault(song_id, len(song_to_code))
        allowed.setflags(write=False)
        song_codes.setflags(write=False)
        return cls(tracks, track_to_code, allowed, song_codes)

    def code(self, track_id: str) -> int:
        return int(self.track_to_code.get(track_id, -1))

    def same_song_mask(self, query_code: int, candidate_codes: np.ndarray) -> np.ndarray:
        query_song = int(self.song_codes[query_code])
        if query_song < 0:
            return np.zeros(len(candidate_codes), dtype=np.bool_)
        return self.song_codes[candidate_codes] == query_song


@dataclass(slots=True)
class EncodedCandidates:
    """One query's union as fixed-width arrays rather than Candidate objects."""

    codec: TrackCodec
    codes: np.ndarray
    source_masks: np.ndarray
    scores: np.ndarray
    _positions: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scores.shape != (len(self.codes), len(SOURCE_NAMES)):
            raise ValueError("compact candidate score matrix has an invalid shape")
        self._positions = {
            int(code): position for position, code in enumerate(self.codes)
        }

    def __len__(self) -> int:
        return len(self.codes)

    def track_id(self, position: int) -> str:
        return self.codec.tracks[int(self.codes[position])]

    def position(self, track_code: int) -> int | None:
        return self._positions.get(track_code)

    def evidence(self, position: int) -> tuple[frozenset[str], dict[str, float]]:
        mask = int(self.source_masks[position])
        sources = frozenset(
            name for index, name in enumerate(SOURCE_NAMES) if mask & (1 << index)
        )
        scores = {
            name: float(self.scores[position, index])
            for index, name in enumerate(SOURCE_NAMES)
            if np.isfinite(self.scores[position, index])
        }
        return sources, scores


@dataclass(frozen=True, slots=True)
class RawVectorBatch:
    """FAISS matrices plus the catalog-code mapping for their index rows."""

    scores: np.ndarray
    rows: np.ndarray
    query_positions: Mapping[int, int]
    row_codes: np.ndarray

    def query(self, batch_position: int) -> tuple[np.ndarray, np.ndarray]:
        search_position = self.query_positions.get(batch_position)
        if search_position is None:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
        rows = self.rows[search_position]
        valid = (rows >= 0) & (rows < len(self.row_codes))
        return self.row_codes[rows[valid]], self.scores[search_position][valid]


@dataclass(frozen=True, slots=True)
class StreamingRecallBatch:
    queries: tuple[str, ...]
    audio: RawVectorBatch
    graph: RawVectorBatch
    bfs_templates: Mapping[str, BfsTemplate]
    tag_templates: Mapping[str, TagTemplate]


@dataclass(frozen=True, slots=True)
class BfsTemplate:
    codes: np.ndarray
    offsets: np.ndarray
    distances: np.ndarray
    similarities: np.ndarray


@dataclass(frozen=True, slots=True)
class TagTemplate:
    recall_codes: np.ndarray
    recall_scores: np.ndarray
    positive_codes: np.ndarray
    positive_scores: np.ndarray


@dataclass(slots=True)
class StreamingRecallEngine:
    """Keep retrain recall numeric until selected pairs are materialized."""

    audio: FaissTrackIndex
    graph: FaissTrackIndex
    bfs: BfsRetriever
    tag: TagRetriever
    codec: TrackCodec
    limits: Mapping[str, int]
    audio_row_codes: np.ndarray = field(init=False, repr=False)
    graph_row_codes: np.ndarray = field(init=False, repr=False)
    bfs_artist_codes: Mapping[str, np.ndarray] = field(init=False, repr=False)
    _bfs_templates: OrderedDict[str, BfsTemplate] = field(
        init=False, repr=False, default_factory=OrderedDict
    )
    _tag_templates: OrderedDict[str, TagTemplate] = field(
        init=False, repr=False, default_factory=OrderedDict
    )

    def __post_init__(self) -> None:
        lookup = self.codec.track_to_code.get
        self.audio_row_codes = np.fromiter(
            (lookup(track_id, -1) for track_id in self.audio.row_to_track),
            dtype=np.int32,
            count=len(self.audio.row_to_track),
        )
        self.graph_row_codes = np.fromiter(
            (lookup(track_id, -1) for track_id in self.graph.row_to_track),
            dtype=np.int32,
            count=len(self.graph.row_to_track),
        )
        self.bfs_artist_codes = {
            artist: np.asarray(
                [
                    code
                    for track_id in sorted(track_ids)
                    if (code := self.codec.code(track_id)) >= 0
                ],
                dtype=np.int32,
            )
            for artist, track_ids in self.bfs.artist_tracks.items()
        }

    def search_many(
        self,
        queries: Sequence[str],
        positive_neighbor_limit: int,
    ) -> StreamingRecallBatch:
        query_tuple = tuple(queries)
        graph_limit = self.limits["graph"] * 3 + 1
        with ThreadPoolExecutor(max_workers=2) as executor:
            audio_job = executor.submit(
                self._search_available,
                self.audio,
                self.audio_row_codes,
                query_tuple,
                positive_neighbor_limit,
            )
            graph_job = executor.submit(
