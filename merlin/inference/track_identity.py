"""Track/song identity used to exclude near-duplicate recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class SameSongFilter:
    track_to_song: Mapping[str, str]

    def __call__(self, left_track_id: str, right_track_id: str) -> bool:
        left = self.track_to_song.get(left_track_id)
        right = self.track_to_song.get(right_track_id)
        return left is not None and right is not None and left == right


def build_track_to_song(rows: Iterable[tuple[str, str | None]]) -> dict[str, str]:
    """Keep valid song IDs and reject conflicting track identities."""
    result: dict[str, str] = {}
    for track_id, song_id in rows:
        if not track_id or not song_id:
            continue
        previous = result.setdefault(track_id, song_id)
        if previous != song_id:
            raise ValueError(f"track {track_id!r} has conflicting song IDs")
    return result


def load_same_song_filter(path: str | Path) -> SameSongFilter:
    """Load track/song identity from prepared metadata in Arrow batches."""
    try:
        import pyarrow.dataset as ds
    except ImportError as error:
        raise RuntimeError("loading track identity requires pyarrow") from error

    dataset = ds.dataset(str(path), format="parquet")

    def rows():
        for batch in dataset.to_batches(columns=["track_id", "song_id"]):
            yield from zip(
                batch.column(0).to_pylist(),
                batch.column(1).to_pylist(),
            )

    return SameSongFilter(build_track_to_song(rows()))
