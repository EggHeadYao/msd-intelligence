# MERLIN Prepare

Prepare raw MSD Parquet files for the MERLIN C1/C2/ranker pipeline.

## Inputs

`prepare.py` expects a raw input directory with the local `parquets/` layout:

- `songs_scalar.parquet`
- `track_metadata.parquet`
- `artist_term.parquet`
- `artist_similarity_edges.parquet`
- `musics/features_*.parquet` or `_extracted/parquets/musics/features_*.parquet`
- `musics/terms_*.parquet` or `_extracted/parquets/musics/terms_*.parquet`

## Outputs

The script writes these prepared tables under the output directory:

- `songs_metadata.parquet`: one row per song with display metadata, `album_key`, and `has_year`.
  **Fields**:
  - `track_id`: MSD track id.
  - `song_id`: MSD song id.
  - `title`: song title.
  - `artist_id`: MSD artist id.
  - `artist_name`: artist display name.
  - `artist_mbid`: MusicBrainz artist id when available.
  - `release`: release or album name.
  - `album_key`: derived album node id, built from `artist_id` and `release`.
  - `duration`: song duration.
  - `year`: release year, with `0` meaning unknown in the raw data.
  - `has_year`: binary flag for whether `year > 0`.
  - `song_hotttnesss`: raw song popularity signal.
  - `artist_hotttnesss`: raw artist popularity signal.
  - `artist_familiarity`: raw artist familiarity signal.
- `song_audio_features_raw.parquet`: one row per song with scalar audio fields and segment aggregate features.
  **Fields**:
  - `track_id`: MSD track id.
  - `danceability`: raw scalar audio feature.
  - `energy`: raw scalar audio feature.
  - `loudness`: raw scalar audio feature.
  - `tempo`: raw scalar audio feature.
  - `duration`: song duration.
  - `key`: estimated musical key.
  - `mode`: estimated major/minor mode.
  - `time_signature`: estimated time signature.
  - `pitch_{mean,std,min,max}_{0..11}`: segment-level pitch aggregates for 12 pitch dimensions.
  - `timbre_{mean,std,min,max}_{0..11}`: segment-level timbre aggregates for 12 timbre dimensions.
  - `loudness_{mean,std,min,max}`: segment-level loudness aggregates.
  - `has_segments`: binary flag for whether segment arrays were available.
- `song_terms.parquet`: `track_id`, `artist_id`, `term` rows for tag-based features and graph edges.
  **Fields**:
  - `track_id`: MSD track id.
  - `artist_id`: MSD artist id attached from metadata.
  - `term`: song-level tag or term.
- `graph_edges.parquet`: typed graph edges for song-artist, song-album, song-tag, artist-tag, song-year, and directed artist-similarity relations.
  **Fields**:
  - `src_type`: source node type, one of `song`, `artist`, `album`, `tag`, or `year`.
  - `src_id`: source node id.
  - `dst_type`: destination node type, one of `song`, `artist`, `album`, `tag`, or `year`.
  - `dst_id`: destination node id.
  - `weight`: fixed edge weight used by graph-based features.
  - `directed`: whether the edge keeps directed semantics.
  - `edge_type`: edge relation type, one of `song_artist`, `song_album`, `song_tag`, `artist_tag`, `song_year`, or `artist_similarity`.

Spark writes each `.parquet` output as a dataset directory containing `part-*.parquet` files.
The output directory is recreated on each run; do not point `--output` at the raw input directory.

## Validate

- Verifies that the prepared output directory contains exactly the expected Parquet tables.
- Checks strict column schemas for metadata, audio features, song terms, and graph edges.
- Validates column data types, including IDs, numeric features, booleans, and graph fields.
- Ensures `songs_metadata` and `song_audio_features_raw` each cover 1,000,000 distinct tracks.
- Confirms metadata and audio feature tables contain the same `track_id` set.
- Checks `has_year` and `has_segments` are binary flags.
- Verifies `song_terms` is non-empty and has no null key fields.
- Checks all required graph edge types exist and have expected counts.
- Ensures artist similarity edges are not duplicated.
- Validates graph node types, edge weights, and directed flags.
- Ensures `song_year` edges do not use `year = 0`.
- Confirms graph edge rows have no null required fields.

## Commands

Run preparation:

```bash
spark-submit --driver-memory 4g p1team02/merlin/prepare/prepare.py --input parquets --output parquets/prepared
```

Validate outputs:

```bash
spark-submit --driver-memory 4g p1team02/merlin/prepare/validate.py --prepared parquets/prepared
```
