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
    artist_codes: np.ndarray | None = None
    artists: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        assignments: Mapping[str, str],
        allowed_splits: frozenset[str],
        same_song: SameSongFilter,
        track_to_artist: Mapping[str, str] | None = None,
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
        artist_to_code: dict[str, int] = {}
        artists: list[str] = []
        artist_codes = np.full(len(tracks), -1, dtype=np.int32)
        for track_id, artist_id in (track_to_artist or {}).items():
            track_code = track_to_code.get(track_id)
            if track_code is not None:
                artist_code = artist_to_code.get(artist_id)
                if artist_code is None:
                    artist_code = len(artists)
                    artist_to_code[artist_id] = artist_code
                    artists.append(artist_id)
                artist_codes[track_code] = artist_code
        allowed.setflags(write=False)
        song_codes.setflags(write=False)
        artist_codes.setflags(write=False)
        return cls(
            tracks,
            track_to_code,
            allowed,
            song_codes,
            artist_codes,
            tuple(artists),
        )

    def code(self, track_id: str) -> int:
        return int(self.track_to_code.get(track_id, -1))

    def same_song_mask(self, query_code: int, candidate_codes: np.ndarray) -> np.ndarray:
        query_song = int(self.song_codes[query_code])
        if query_song < 0:
            return np.zeros(len(candidate_codes), dtype=np.bool_)
        return self.song_codes[candidate_codes] == query_song

    def same_artist_mask(
        self,
        query_code: int,
        candidate_codes: np.ndarray,
    ) -> np.ndarray:
        if self.artist_codes is None:
            return np.zeros(len(candidate_codes), dtype=np.bool_)
        query_artist = int(self.artist_codes[query_code])
        if query_artist < 0:
            return np.zeros(len(candidate_codes), dtype=np.bool_)
        return self.artist_codes[candidate_codes] == query_artist


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
    backfill_limits: Mapping[str, int]
    backfill_order: tuple[str, ...]
    candidate_limit: int
    audio_row_codes: np.ndarray = field(init=False, repr=False)
    audio_code_rows: np.ndarray = field(init=False, repr=False)
    graph_row_codes: np.ndarray = field(init=False, repr=False)
    bfs_artist_codes: Mapping[str, np.ndarray] = field(init=False, repr=False)
    _bfs_templates: OrderedDict[str, BfsTemplate] = field(
        init=False, repr=False, default_factory=OrderedDict
    )
    _tag_templates: OrderedDict[str, TagTemplate] = field(
        init=False, repr=False, default_factory=OrderedDict
    )

    def __post_init__(self) -> None:
        if set(self.limits) != set(SOURCE_NAMES) or set(
            self.backfill_limits
        ) != set(SOURCE_NAMES):
            raise ValueError("streaming recall limits must cover all sources")
        if any(
            self.backfill_limits[name] < self.limits[name]
            for name in SOURCE_NAMES
        ):
            raise ValueError("streaming backfill cannot reduce a primary quota")
        if set(self.backfill_order) - set(SOURCE_NAMES):
            raise ValueError("streaming backfill order contains an unknown source")
        if self.candidate_limit < sum(self.limits.values()):
            raise ValueError("streaming candidate limit is below primary quotas")
        lookup = self.codec.track_to_code.get
        self.audio_row_codes = np.fromiter(
            (lookup(track_id, -1) for track_id in self.audio.row_to_track),
            dtype=np.int32,
            count=len(self.audio.row_to_track),
        )
        self.audio_code_rows = np.full(len(self.codec.tracks), -1, dtype=np.int64)
        valid_audio_rows = self.audio_row_codes >= 0
        self.audio_code_rows[self.audio_row_codes[valid_audio_rows]] = np.flatnonzero(
            valid_audio_rows
        )
        self.audio_code_rows.setflags(write=False)
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
                self._search_available,
                self.graph,
                self.graph_row_codes,
                query_tuple,
                graph_limit,
            )
            audio = audio_job.result()
            graph = graph_job.result()
        bfs_templates = self._prepare_bfs_templates(query_tuple)
        tag_templates = self._prepare_tag_templates(query_tuple)
        return StreamingRecallBatch(
            query_tuple,
            audio,
            graph,
            bfs_templates,
            tag_templates,
        )

    def search_candidates_many(
        self,
        queries: Sequence[str],
    ) -> StreamingRecallBatch:
        """Run the canonical over-fetched searches needed by candidate export."""
        return self.search_many(queries, self.limits["audio"] * 3 + 1)

    @staticmethod
    def _search_available(
        index: FaissTrackIndex,
        row_codes: np.ndarray,
        queries: tuple[str, ...],
        limit: int,
    ) -> RawVectorBatch:
        positions = [position for position, query in enumerate(queries) if index.contains(query)]
        available = [queries[position] for position in positions]
        scores, rows = index.search_many_raw(available, limit)
        return RawVectorBatch(
            scores,
            rows,
            {
                batch_position: search_position
                for search_position, batch_position in enumerate(positions)
            },
            row_codes,
        )

    def query(
        self,
        batch: StreamingRecallBatch,
        position: int,
    ) -> tuple[EncodedCandidates, list[tuple[str, float]], list[tuple[str, float]]]:
        query_id = batch.queries[position]
        query_code = self.codec.code(query_id)
        audio_codes, audio_scores = batch.audio.query(position)
        graph_codes, graph_scores = batch.graph.query(position)
        audio_codes, audio_scores = self._ordered(audio_codes, audio_scores)
        graph_codes, graph_scores = self._ordered(graph_codes, graph_scores)
        audio_group = self._filter_group(
            query_code, audio_codes, audio_scores, self.backfill_limits["audio"]
        )
        graph_group = self._filter_group(
            query_code, graph_codes, graph_scores, self.backfill_limits["graph"]
        )
        root_artist = self.bfs.track_to_artist.get(query_id)
        bfs_group = self._bfs_group(
            query_code,
            batch.bfs_templates.get(root_artist) if root_artist is not None else None,
        )
        root_artist = self.tag.track_to_artist.get(query_id)
        tag_group, tag_positive = self._tag_group(
            query_code,
            batch.tag_templates.get(root_artist) if root_artist is not None else None,
        )
        groups = self._allocate_groups((
            audio_group, graph_group, bfs_group, tag_group
        ))
        candidates = self._merge(groups)
        audio_positive = [
            (self.codec.tracks[int(code)], float(score))
            for code, score in zip(audio_codes, audio_scores, strict=True)
            if code >= 0
        ]
        tag_positive_rows = [
            (self.codec.tracks[int(code)], float(score))
            for code, score in zip(*tag_positive, strict=True)
        ]
        return candidates, audio_positive, tag_positive_rows

    def candidate_query(
        self,
        batch: StreamingRecallBatch,
        position: int,
    ) -> tuple[EncodedCandidates, RecallAudit]:
        """Build the canonical candidate union without positive-pool materialization."""
        query_id = batch.queries[position]
        query_code = self.codec.code(query_id)
        audio_codes, audio_scores = batch.audio.query(position)
        graph_codes, graph_scores = batch.graph.query(position)
        audio_codes, audio_scores = self._ordered(audio_codes, audio_scores)
        graph_codes, graph_scores = self._ordered(graph_codes, graph_scores)
        expanded_groups = (
            self._filter_group(
                query_code,
                audio_codes,
                audio_scores,
                self.backfill_limits["audio"],
            ),
            self._filter_group(
                query_code,
                graph_codes,
                graph_scores,
                self.backfill_limits["graph"],
            ),
            self._bfs_group(
                query_code,
                batch.bfs_templates.get(self.bfs.track_to_artist.get(query_id)),
            ),
            self._tag_group(
                query_code,
                batch.tag_templates.get(self.tag.track_to_artist.get(query_id)),
            )[0],
        )
        groups = self._allocate_groups(expanded_groups)
        candidates = self._merge(groups)
        counts = {
            name: len(group[0]) for name, group in zip(SOURCE_NAMES, groups, strict=True)
        }
        raw_count = sum(counts.values())
        unique_count = len(candidates)
        exclusive = {
            name: int(np.count_nonzero(candidates.source_masks == (1 << index)))
            for index, name in enumerate(SOURCE_NAMES)
        }
        availability = {
            "audio": self.audio.contains(query_id),
            "graph": self.graph.contains(query_id),
            "bfs": self.bfs.is_available(query_id),
            "tag": self.tag.is_available(query_id),
        }
        return candidates, RecallAudit(
            source_counts=counts,
            source_shortages={
                name: max(0, int(self.limits[name]) - count)
                for name, count in counts.items()
            },
            unique_candidates=unique_count,
            raw_candidates=raw_count,
            duplicate_candidates=raw_count - unique_count,
            deduplication_rate=(raw_count - unique_count) / raw_count if raw_count else 0.0,
            exclusive_candidates=exclusive,
            source_available=availability,
        )

    def _ordered(
        self,
        codes: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = codes >= 0
        codes = codes[valid]
        scores = scores[valid]
        order = np.lexsort((codes, -scores))
        return codes[order], scores[order]

    def _filter_group(
        self,
        query_code: int,
        codes: np.ndarray,
        scores: np.ndarray,
        limit: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = (
            (codes != query_code)
            & self.codec.allowed[codes]
            & ~self.codec.same_song_mask(query_code, codes)
        )
        codes = codes[valid]
        scores = scores[valid]
        if len(codes):
            _, first = np.unique(codes, return_index=True)
            keep = np.sort(first)[:limit]
            codes = codes[keep]
            scores = scores[keep]
        return codes.astype(np.int32, copy=False), scores.astype(np.float32, copy=False)

    def _prepare_bfs_templates(
        self,
        queries: Sequence[str],
    ) -> Mapping[str, BfsTemplate]:
        roots = tuple(dict.fromkeys(
            root
            for query_id in queries
            if (root := self.bfs.track_to_artist.get(query_id)) is not None
        ))
        prepared = {
            root: template
            for root in roots
            if (template := self._bfs_templates.get(root)) is not None
        }
        missing = [root for root in roots if root not in prepared]
        reachable_by_root = {
            root: tuple(
                (artist, distance)
                for artist, distance in self.bfs._distances(root).items()
                if distance != 0
            )
            for root in missing
        }
        scores_by_root = self.tag.artist_similarities_many({
            root: [artist for artist, _distance in reachable]
            for root, reachable in reachable_by_root.items()
        })
        for root in missing:
            template = self._build_bfs_template(
                reachable_by_root[root],
                scores_by_root[root],
            )
            prepared[root] = template
            self._remember(self._bfs_templates, root, template, 512)
        return prepared

    def _build_bfs_template(
        self,
        reachable: Sequence[tuple[str, int]],
        tag_scores: Sequence[float],
    ) -> BfsTemplate:
        artist_codes: list[np.ndarray] = []
        offsets = [0]
        distances: list[int] = []
        for (artist, distance), _similarity in zip(
            reachable,
            tag_scores,
            strict=True,
        ):
            codes = self.bfs_artist_codes.get(
                artist,
                np.empty(0, dtype=np.int32),
            )
            artist_codes.append(codes)
            offsets.append(offsets[-1] + len(codes))
            distances.append(distance)
        return BfsTemplate(
            np.concatenate(artist_codes) if artist_codes else np.empty(0, dtype=np.int32),
            np.asarray(offsets, dtype=np.int32),
            np.asarray(distances, dtype=np.uint8),
            np.asarray(tag_scores, dtype=np.float32),
        )

    def _bfs_group(
        self,
        query_code: int,
        template: BfsTemplate | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if template is None:
            return self._empty_group()
        groups: list[np.ndarray] = []
        distances: list[np.ndarray] = []
        similarities: list[np.ndarray] = []
        query_song = int(self.codec.song_codes[query_code])
        for index, (distance, similarity) in enumerate(
            zip(template.distances, template.similarities, strict=True)
        ):
            codes = template.codes[template.offsets[index] : template.offsets[index + 1]]
            selected = (
                codes
                if query_song < 0
                else codes[self.codec.song_codes[codes] != query_song]
            )[: self.bfs.per_artist_cap]
            if len(selected):
                groups.append(selected)
                distances.append(
                    np.full(len(selected), int(distance), dtype=np.uint8)
                )
                similarities.append(
                    np.full(len(selected), float(similarity), dtype=np.float32)
                )
        if not groups:
            return self._empty_group()
        codes = np.concatenate(groups)
        hops = np.concatenate(distances)
        tag_scores = np.concatenate(similarities)
        order = np.lexsort((codes, -tag_scores, hops))[: self.limits["bfs"]]
        return (
            codes[order].astype(np.int32, copy=False),
            (1.0 / (1.0 + hops[order])).astype(np.float32, copy=False),
        )

    def _prepare_tag_templates(
        self,
        queries: Sequence[str],
    ) -> Mapping[str, TagTemplate]:
        roots = tuple(dict.fromkeys(
            root
            for query_id in queries
            if (root := self.tag.track_to_artist.get(query_id)) is not None
        ))
        prepared = {
            root: template
            for root in roots
            if (template := self._tag_templates.get(root)) is not None
        }
        missing = [root for root in roots if root not in prepared]
        neighbors = self.tag.similar_artists_many(
            missing,
            self.tag.artist_neighbor_limit,
        )
        for root in missing:
            template = self._build_tag_template(neighbors.get(root, ()))
            prepared[root] = template
            self._remember(self._tag_templates, root, template, 512)
        return prepared

    def _build_tag_template(
        self,
        artist_neighbors: Sequence[tuple[str, float]],
    ) -> TagTemplate:
        recall: list[tuple[int, float]] = []
        positives: list[tuple[int, float]] = []
        for artist, score in artist_neighbors:
            tracks = self.tag.artist_tracks.get(artist, ())
            recall.extend(
                (self.codec.code(track_id), float(score))
                for track_id in tracks[: self.tag.per_artist_cap]
            )
            positives.extend(
                (self.codec.code(track_id), float(score))
                for track_id in sorted(tracks)
            )
        recall = [row for row in recall if row[0] >= 0]
        positives = [row for row in positives if row[0] >= 0]
        return TagTemplate(
            np.asarray([row[0] for row in recall], dtype=np.int32),
            np.asarray([row[1] for row in recall], dtype=np.float32),
            np.asarray([row[0] for row in positives], dtype=np.int32),
            np.asarray([row[1] for row in positives], dtype=np.float32),
        )

    def _tag_group(
        self,
        query_code: int,
        template: TagTemplate | None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        if template is None:
            empty = self._empty_group()
            return empty, empty
        recall = self._filter_group(
            query_code,
            template.recall_codes,
            template.recall_scores,
            self.limits["tag"],
        )
        return recall, (template.positive_codes, template.positive_scores)

    def _merge(
        self,
        groups: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> EncodedCandidates:
        positions: dict[int, int] = {}
        codes: list[int] = []
        masks: list[int] = []
        score_rows: list[list[float]] = []
        for source_index, (group_codes, group_scores) in enumerate(groups):
            for code, score in zip(group_codes, group_scores, strict=True):
                value = int(code)
                position = positions.get(value)
                if position is None:
                    position = len(codes)
                    positions[value] = position
                    codes.append(value)
                    masks.append(0)
                    score_rows.append([float("nan")] * len(SOURCE_NAMES))
                masks[position] |= 1 << source_index
                score_rows[position][source_index] = float(score)
        return EncodedCandidates(
            self.codec,
            np.asarray(codes, dtype=np.int32),
            np.asarray(masks, dtype=np.uint8),
            np.asarray(score_rows, dtype=np.float32).reshape((-1, len(SOURCE_NAMES))),
        )

    def _allocate_groups(
        self,
        groups: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Retain primary groups and round-robin unused capacity."""
        selected_ends = [
            min(len(group[0]), self.limits[name])
            for name, group in zip(SOURCE_NAMES, groups, strict=True)
        ]
        used = {
            int(code)
            for source, (codes, _scores) in enumerate(groups)
            for code in codes[: selected_ends[source]]
        }
        source_positions = {name: index for index, name in enumerate(SOURCE_NAMES)}
        while len(used) < self.candidate_limit:
            progressed = False
            for name in self.backfill_order:
                source = source_positions[name]
                codes, scores = groups[source]
                boundary = min(len(codes), self.backfill_limits[name])
                while selected_ends[source] < boundary:
                    position = selected_ends[source]
                    selected_ends[source] += 1
                    code = int(codes[position])
                    if code not in used:
                        used.add(code)
                        progressed = True
                        break
                if len(used) >= self.candidate_limit:
                    break
            if not progressed:
                break
        return tuple(
            (
                codes[: selected_ends[source]],
                scores[: selected_ends[source]],
            )
            for source, (codes, scores) in enumerate(groups)
        )

    @staticmethod
    def _remember(cache: OrderedDict, key: str, value: object, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > limit:
            cache.popitem(last=False)

    @staticmethod
    def _empty_group() -> tuple[np.ndarray, np.ndarray]:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
