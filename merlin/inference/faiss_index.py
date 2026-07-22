"""Pure-Python access to a FAISS index keyed by MERLIN track IDs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_lineage import load_faiss_manifest


@dataclass(slots=True)
class FaissTrackIndex:
    """Bind a FAISS row index to its stable ``track_id`` mapping."""

    index: Any
    row_to_track: tuple[str, ...]
    _track_to_row: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.index.ntotal) != len(self.row_to_track):
            raise ValueError("FAISS index size does not match track-id mapping")
        if not self.row_to_track or any(not track_id for track_id in self.row_to_track):
            raise ValueError("track-id mapping must be non-empty")
        self._track_to_row = {
            track_id: row_id for row_id, track_id in enumerate(self.row_to_track)
        }
        if len(self._track_to_row) != len(self.row_to_track):
            raise ValueError("track-id mapping contains duplicate track IDs")

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    def contains(self, track_id: str) -> bool:
        return track_id in self._track_to_row

    def similarity(self, left_track_id: str, right_track_id: str) -> float | None:
        """Return catalog cosine, or ``None`` when either embedding is absent."""
        if not self.contains(left_track_id) or not self.contains(right_track_id):
            return None
        left = self._reconstruct(left_track_id)
        right = self._reconstruct(right_track_id)
        score = float(np.dot(left, right))
        if not np.isfinite(score):
            raise ValueError("FAISS pair similarity is not finite")
        return score

    def _reconstruct(self, track_id: str) -> np.ndarray:
        row_id = self._track_to_row[track_id]
        cached = self._vector_cache.get(row_id)
        if cached is not None:
            self._vector_cache.move_to_end(row_id)
            return cached
        vector = np.asarray(
            self.index.reconstruct(row_id),
            dtype=np.float32,
        )
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError("reconstructed FAISS vector is invalid")
        if abs(float(np.linalg.norm(vector)) - 1.0) > 1e-5:
            raise ValueError("reconstructed FAISS vector is not unit normalized")
        vector.setflags(write=False)
        self._vector_cache[row_id] = vector
        if len(self._vector_cache) > self._VECTOR_CACHE_SIZE:
            self._vector_cache.popitem(last=False)
        return vector

    @classmethod
    def from_files(
        cls,
        index_path: str | Path,
        mapping_path: str | Path,
        manifest_path: str | Path,
        encoder_metadata_path: str | Path,
        *,
        expected_space: str,
        expected_contract_key: str,
        expected_contract: str,
    ) -> "FaissTrackIndex":
        """Load a manifest-bound C1/C2 index and Parquet row mapping."""
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError("loading a FAISS index requires the faiss package") from error
        index_file = Path(index_path)
        if not index_file.is_file():
            raise FileNotFoundError(f"FAISS index does not exist: {index_file}")
        manifest = load_faiss_manifest(
            manifest_path,
            index_path=index_file,
            mapping_path=mapping_path,
            encoder_metadata_path=encoder_metadata_path,
            expected_space=expected_space,
            expected_contract_key=expected_contract_key,
            expected_contract=expected_contract,
        )
        index = faiss.read_index(str(index_file))
        if type(index).__name__ != "IndexFlatIP" or int(index.d) != 128:
            raise ValueError("FAISS artifact must be an exact 128D IndexFlatIP")
        result = cls(index=index, row_to_track=_load_track_mapping(mapping_path))
        if int(manifest["row_count"]) != int(index.ntotal):
            raise ValueError("FAISS manifest row count mismatch")
        return result

    def search(self, query_track_id: str, limit: int) -> list[tuple[str, float]]:
        """Return nearest tracks ordered by FAISS inner-product score."""
        if limit <= 0:
            raise ValueError("FAISS search limit must be positive")
        if not self.contains(query_track_id):
            raise KeyError(f"query track is not in FAISS mapping: {query_track_id}")
        query = self._reconstruct(query_track_id)
        return self.search_vector(query, limit)

    def search_vector(
        self,
        query_embedding: Any,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Search a cold 128D C1 vector after safe float32 normalization."""
        if limit <= 0:
            raise ValueError("FAISS search limit must be positive")
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise ValueError(f"query embedding must have shape ({self.dimension},)")
        if not np.all(np.isfinite(query)):
            raise ValueError("query embedding contains NaN or infinite values")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("query embedding must have a finite non-zero norm")
        query = (query / norm).astype(np.float32, copy=False).reshape(1, -1)
        result_limit = min(limit, int(self.index.ntotal))
        search_engine = os.environ.get("MERLIN_FAISS_SEARCH_ENGINE", "faiss")
        if search_engine == "faiss":
            scores, row_ids = self.index.search(query, result_limit)
        elif search_engine == "numpy":
            scores, row_ids = self._numpy_exact_search(query[0], result_limit)
        else:
            raise ValueError("MERLIN_FAISS_SEARCH_ENGINE must be faiss or numpy")
        results = [
            (self.row_to_track[int(result_row)], float(score))
            for result_row, score in zip(row_ids[0], scores[0], strict=True)
            if 0 <= int(result_row) < len(self.row_to_track)
        ]
        return sorted(results, key=lambda item: (-item[1], item[0]))

    def _numpy_exact_search(
        self,
        query: np.ndarray,
        limit: int,
        *,
        block_size: int = 50_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility exact-IP search for FAISS wheels with unsupported SIMD."""
        candidates: list[tuple[float, int]] = []
        total = int(self.index.ntotal)
        for start in range(0, total, block_size):
            count = min(block_size, total - start)
            matrix = np.asarray(
                self.index.reconstruct_n(start, count),
                dtype=np.float32,
            )
            block_scores = matrix @ query
            if not np.all(np.isfinite(block_scores)):
                raise ValueError("FAISS reconstructed block contains non-finite scores")
            local_limit = min(limit, count)
            local_rows = np.argpartition(block_scores, -local_limit)[-local_limit:]
            candidates.extend(
                (float(block_scores[row]), start + int(row)) for row in local_rows
            )
        best = sorted(
            candidates,
            key=lambda item: (-item[0], self.row_to_track[item[1]]),
        )[:limit]
        return (
            np.asarray([[score for score, _row in best]], dtype=np.float32),
            np.asarray([[row for _score, row in best]], dtype=np.int64),
        )


def _load_track_mapping(mapping_path: str | Path) -> tuple[str, ...]:
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS track mapping does not exist: {path}")
    tracks: list[str] = []
    for expected_row, (row_id, track_id) in enumerate(
        parquet_rows(
            path,
            ("row_id", "track_id"),
            order_by=("row_id",),
        )
    ):
        if row_id is None or track_id is None:
            raise ValueError("FAISS track mapping contains null values")
        if int(row_id) != expected_row:
            raise ValueError("FAISS mapping row_id must be contiguous from zero")
        tracks.append(str(track_id))
    return tuple(tracks)
