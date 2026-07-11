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
- `song_audio_features_raw.parquet`: one row per song with scalar audio fields and segment aggregate features.
- `song_terms.parquet`: `track_id`, `artist_id`, `term` rows for tag-based features and graph edges.
- `graph_edges.parquet`: typed graph edges for song-artist, song-album, song-tag, artist-tag, song-year, and directed artist-similarity relations.

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
spark-submit --driver-memory 4g p1team02/tools/merlin/prepare/prepare.py --input parquets --output parquets/prepared
```

Validate outputs:

```bash
spark-submit --driver-memory 4g p1team02/tools/merlin/prepare/validate.py --prepared parquets/prepared
```
