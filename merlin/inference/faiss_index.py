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
        vector = np.asarray(
            self.index.reconstruct(self._track_to_row[track_id]),
            dtype=np.float32,
        )
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError("reconstructed FAISS vector is invalid")
        if abs(float(np.linalg.norm(vector)) - 1.0) > 1e-5:
            raise ValueError("reconstructed FAISS vector is not unit normalized")
        return vector

    @classmethod
    def from_files(
        cls,
        index_path: str | Path,
        mapping_path: str | Path,
        manifest_path: str | Path,
        *,
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
        scores, row_ids = self.index.search(query, min(limit, int(self.index.ntotal)))
        results = [
            (self.row_to_track[int(result_row)], float(score))
            for result_row, score in zip(row_ids[0], scores[0], strict=True)
            if 0 <= int(result_row) < len(self.row_to_track)
        ]
        return sorted(results, key=lambda item: (-item[1], item[0]))


def _load_track_mapping(mapping_path: str | Path) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("loading a track mapping requires the pyarrow package") from error
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS track mapping does not exist: {path}")
    table = parquet.read_table(path, columns=["row_id", "track_id"])
    pairs = sorted(zip(table["row_id"].to_pylist(), table["track_id"].to_pylist()))
    if any(row_id is None or track_id is None for row_id, track_id in pairs):
        raise ValueError("FAISS track mapping contains null values")
    if [int(row_id) for row_id, _ in pairs] != list(range(len(pairs))):
        raise ValueError("FAISS mapping row_id must be contiguous from zero")
    return tuple(str(track_id) for _, track_id in pairs)
