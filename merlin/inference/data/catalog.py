"""One-pass catalog metadata loading shared by C3 batch stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .features_v2 import TrackMetadataV2
from .parquet_io import parquet_rows
from .tag_data import TagData, build_tag_data
from .track_identity import SameSongFilter


@dataclass(frozen=True, slots=True)
class CatalogContext:
    same_song: SameSongFilter
    tag_data: TagData
    ranker_tracks: Mapping[str, TrackMetadataV2]


def load_catalog_context(
    songs_metadata_path: str | Path,
    graph_edges_path: str | Path,
    *,
    include_ranker_metadata: bool = False,
) -> CatalogContext:
    """Project songs metadata once and construct all shared in-memory mappings."""
    columns = ["track_id", "song_id", "artist_id"]
    if include_ranker_metadata:
        columns.extend(("release_7digitalid", "year", "has_year"))
    track_to_song: dict[str, str] = {}
    track_to_artist: dict[str, str] = {}
    ranker_tracks: dict[str, TrackMetadataV2] = {}
    for values in parquet_rows(songs_metadata_path, columns):
        track_id = str(values[0]) if values[0] else ""
        if not track_id:
            continue
        song_id = str(values[1]) if values[1] else None
        artist_id = str(values[2]) if values[2] else None
        if song_id is not None:
            previous_song = track_to_song.setdefault(track_id, song_id)
            if previous_song != song_id:
                raise ValueError(f"track {track_id!r} has conflicting song IDs")
        if artist_id is not None:
            previous_artist = track_to_artist.setdefault(track_id, artist_id)
            if previous_artist != artist_id:
                raise ValueError(f"track {track_id!r} has multiple artists")
        if include_ranker_metadata:
            release_id, year, has_year = values[3:]
            metadata = TrackMetadataV2(
                release_id=(
                    str(release_id)
                    if release_id not in (None, "", 0, "0")
                    else None
                ),
                year=int(year) if has_year and year is not None else None,
            )
            previous_metadata = ranker_tracks.setdefault(track_id, metadata)
            if previous_metadata != metadata:
                raise ValueError(f"track {track_id!r} has conflicting v2 metadata")

    terms = parquet_rows(
        graph_edges_path,
        ("src_id", "dst_id"),
        edge_type="artist_term",
    )
    return CatalogContext(
        same_song=SameSongFilter(track_to_song),
        tag_data=build_tag_data(track_to_artist.items(), terms),
        ranker_tracks=ranker_tracks,
    )
