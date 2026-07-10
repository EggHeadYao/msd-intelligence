# HDF5 Extraction Tools

Extract Million Song Dataset HDF5 files into Parquet for Drill, Spark, and the MERLIN recommender.

## Scripts

### Code Quality

Both two scripts are checked by `pyrefly` and `ruff`.

### `extract_summary.py`

Reads scalar fields from `msd_summary_file.h5` (a single 301 MB file) and writes one Parquet file.

```bash
python extract_summary.py <msd_summary_file.h5> <output.parquet>
```

Output: 1 000 000 rows x 18 columns (track_id, danceability, energy, loudness, tempo, duration, key, mode, time_signature, artist_id, artist_name, release, song_id, song_hotttnesss, artist_hotttnesss, artist_familiarity, title, year).  Runs in ~30 s on the SSHFS mount.

### `extract_musics.py`

Walks the per-song `.h5` directory, extracts 100-dim aggregated segment features (pitch/timbre/loudness mean/std/min/max) plus a `has_segments` binary mask, similar-artist pairs, and artist-term pairs, writing them in batch Parquet files.

```bash
python extract_musics.py <data_root> <output_dir>
python extract_musics.py <data_root> <output_dir> -w 8   # 8 workers
```

Output per batch:
- `features_NNNN.parquet` -- N rows x 102 columns (track_id + 100
  feature columns + has_segments).
- `similar_NNNN.parquet` -- (track_id, similar_artist_id) pairs,
  empty strings filtered.
- `terms_NNNN.parquet` -- (track_id, term) pairs.

The `features` files track_id enables direct join with the scalar Parquet without relying on row position.  `has_segments` allows the Spark post-processing step to fill missing segments with the training-set mean instead of literal zeros.

Checkpoint/resume is supported via a line-delimited `checkpoint.txt`. Restarting with the same output directory skips already-processed files and continues from the next batch index.

## 10K Cold-Cache Benchmark

Two sample of 10 000 `.h5` files was drawn from the dataset and processed with both a single worker and 8 parallel workers. 

### Timing

| Workers | Wall-clock | Notes |
|--------:|:---------- |:------|
| 1       | ~9 min 24 s | Sequential |
| 8       | **5 min 35 s** | `imap_unordered` on `multiprocessing.Pool` |

### Bottleneck Analysis

Per-file profile (cold cache, SSHFS mount):

| Operation                       | p50 latency |
|---------------------------------|:-----------:|
| `open()` via SSHFS              | ~4.5 ms |
| Read 3 HDF5 datasets            | ~8.5 ms |
| Segment aggregation (numpy)     | ~30 ms   |
| String decode + Python overhead | ~13 ms   |
| **Total per file**              | **~56 ms** |

The dominant cost is segment aggregation (numpy mean/std/min/max on ~800x12 arrays) followed by SSHFS I/O.  Neither is improved by switching languages -- numpy is already C-optimised, and the I/O path is gated by network latency to the remote server.

### Correctness

- All 10 000 output rows have 0 nulls, 0 NaN, 0 Inf, and 0 zero-variance columns.
- Three randomly selected files were cross-checked against their original HDF5 sources: `pitch_mean_0` matched to 6 decimal places; similar-artist and term counts matched exactly.
- Multi-batch output, 8-worker parallelism, and checkpoint resume were verified on a 1 000-file sample (10 batches of 100).

### Full 1M Projection

Extrapolating from the 10K timing:

| Workers | Estimated wall-clock |
|--------:|:---------------------|
| 8       | ~9 hours             |

This is a one-time extraction.  With checkpoint/resume the job can be split across multiple sessions.
