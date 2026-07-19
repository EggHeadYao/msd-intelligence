# HDF5 Extraction Tools

Extract Million Song Dataset HDF5 files into Parquet for Drill, Spark, and the MERLIN recommender.

## Scripts

### Code Quality

The extraction scripts are checked by `pyrefly` and `ruff`.

### `extract_summary.py`

Reads the selected scalar fields from `msd_summary_file.h5` into memory and writes one Parquet file with explicit missing-value semantics.

```bash
python extract_summary.py \
  millionsong/AdditionalFiles/msd_summary_file.h5 \
  <output_dir>
```

The fixed 23-column output is:

```text
track_id, loudness, tempo, duration
key, key_confidence, mode, mode_confidence
time_signature, time_signature_confidence
end_of_fade_in, start_of_fade_out
artist_id, artist_name, release, release_7digitalid
song_id, song_hotttnesss, artist_hotttnesss, artist_familiarity
title, track_7digitalid, year
```

`danceability` and `energy` are intentionally absent because the MSD values are all zero. `title`, `artist_name`, and `release` remain as display metadata and are not C1 model features. Non-positive ID/year/tempo/time-signature sentinels and non-finite floats become Parquet nulls.

### `extract_musics.py`

Walks the 1M per-track `.h5` tree and writes one fixed 615-dimensional shared audio-array summary per track. It supports multiprocessing, batch Parquet output, error reporting, and checkpoint/resume. MERLIN and Year Prediction project their own model views from this one shared table; the extractor does not rescan HDF5 for either model.

```bash
python extract_musics.py \
  millionsong/data \
  <output_dir> \
  -w 8 \
  --batch-size 10000
```

The production batch size is 10,000. Each `features_NNNN.parquet` contains `track_id` plus 615 nullable float64 features:

- The original ordered 308-column prefix: 112 robust summaries, 100 quarter-pooling values, 40 segment timing/delta values, 45 rhythm/structure values, and 11 availability/quality values.
- 28 duration-weighted q50 values and 50 half-pooling values.
- The official paper-aligned T90 block: 12 raw segment-timbre means plus 78 sample-covariance values in diagonal-offset order.
- 60 global and 72 half/quarter key-relative pitch values.
- Two pitch-profile shape values and five additional availability masks.

`audio_features.py` defines the ordered 615-column schema, the 539-column MERLIN projection, and all pure NumPy aggregation formulas. `extract_musics.py` handles HDF5 I/O, workers, nullable Arrow output, batch commits, recovery, and the CLI. Missing derived values are Parquet nulls rather than ordinary zeros; masks remain non-null float64 values in `{0.0, 1.0}`.

`feature_contract.json` freezes `shared_audio_615_v1`, the input root, feature count, full column order, and its SHA-256 hash. `checkpoint.txt` stores relative HDF5 paths. Restarting the exact same command skips completed tracks and continues at the next batch index. Existing Parquet schemas and track IDs are reconciled into the checkpoint, so interruption after a batch rename but before checkpoint replacement does not duplicate rows. A contract, input-root, or schema mismatch fails closed. Failed files are not checkpointed, are recorded in the current invocation's `errors.txt`, and make the command exit nonzero; fix the source/mount issue and rerun the same command. A clean retry removes the stale error file.

`--batch-size` and `--limit` are test controls. `--limit N` processes at most N remaining files in that invocation; omit both options for the production run. Always use a dedicated output directory. Existing batches without a matching `feature_contract.json` are rejected.

### `extract_chores.py`

Exports supplementary MSD files (SQLite databases and delimited text
tables) to Parquet for cross-validation and alternative data views.

```bash
python extract_chores.py <AdditionalFiles_dir> <output_dir>
```

Output:

| Source                | Output                    |      Rows |
| --------------------- | ------------------------- | --------: |
| `track_metadata.db`   | `track_metadata.parquet`  | 1 000 000 |
| `artist_term.db`      | `artist_term.parquet`     | 1 109 381 |
| `artist_term.db`      | `artist_mbtag.parquet`    |    24 777 |
| `tracks_per_year.txt` | `tracks_per_year.parquet` |   515 576 |
| `artist_location.txt` | `artist_location.parquet` |    13 850 |

The following files are intentionally skipped:

- `artist_similarity.db` -- already converted by the Java `convert`
  module during P1M1.
- `unique_artists.txt`, `unique_tracks.txt`, `unique_terms.txt`,
  `unique_mbtags.txt` -- pure identifier lists already derivable from
  the extracted Parquet tables (e.g. `SELECT DISTINCT artist_id`
  yields the same information).  Converting them to Parquet adds no
  value beyond what `wc -l` provides.

## Validation and Reference Performance

The final 615-column implementation was validated on 5,000 mounted MSD tracks with 8 workers: 5,000 unique IDs, 616 exact columns including `track_id`, no failed files, no infinities, binary non-null masks, contract-consistent nulls, and an exact checkpoint. An independently recomputed real-track T90 vector matched all 90 stored values exactly. Full-run speed remains dominated by mounted HDF5 I/O and should not be inferred from one cached small run.
