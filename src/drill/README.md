# Apache Drill Queries

This directory contains the four Apache Drill queries required by the project: the valid song-year range, the multi-criteria extreme song, the album with the most distinct tracks, and the artist who recorded the longest song.

## Data

The queries use `songs_scalar.parquet` and the projected `energy` column from `audio_features/features_*.parquet`. The default data directory is `parquets/year_prediction/raw` under the repository root. Pass another directory as the first argument to the runner when the same Parquet layout is stored elsewhere.

The Drill storage override exposes the selected directory as the read-only `dfs.msd` workspace. Query files therefore contain no machine-specific absolute paths.

## Run

Apache Drill 1.22.0 is expected under `/usr/local/drill` unless `DRILL_HOME` is set.

```bash
./src/drill/scripts/run_all.sh
```

Use a different Parquet directory with:

```bash
./src/drill/scripts/run_all.sh /path/to/year_prediction/raw
```

Each query starts an embedded Drill session and writes a separate CSV file under `results/`.

## Current Results

- Valid song years range from 1922 to 2011.
- The lexicographic extreme-song query returns "Jingle Bell Rock" by Bobby Helms; its extracted energy is unavailable.
- "First Time In A Long Time: The Reprise Recordings" has the largest album count with 85 distinct tracks.
- "Grounation" by Mystic Revelation of Rastafari is the longest track at 3034.90567 seconds.

## Query Semantics

- `01_year_range.sql` ignores `year = 0`, which denotes an unknown MSD release year.
- `02_extreme_song.sql` applies the requested criteria lexicographically: hotness descending, duration ascending, energy descending, and tempo ascending. The extracted MSD energy column is missing for all one million tracks, so the query reports `energy_available = false` and uses `0.0` only as a neutral deterministic sort value.
- `03_album_most_tracks.sql` groups by `release_7digitalid` instead of album name so unrelated albums with the same title are not merged.
- `04_longest_song_artist.sql` reports the track and its `artist_name`; MSD uses this field for both bands and individual artists.
