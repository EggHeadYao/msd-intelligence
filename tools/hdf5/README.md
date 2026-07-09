# HDF5 Extraction Tools

Extract Million Song Dataset HDF5 files into Parquet for Drill, Spark, and the MERLIN recommender.

## Scripts

### Code Quality

Both two scripts are checked by `pyrefly` and `ruff`.

### `extract_summary.py`

Reads the scalar fields from `msd_summary_file.h5` (a single 301 MB file bundling 1M songs' worth of Echo Nest analysis, metadata, and MusicBrainz data) and writes one Parquet file.

```bash
python extract_summary.py <msd_summary_file.h5> <output.parquet>
```

Output: 1 000 000 rows x 17 columns (track_id, danceability, energy, loudness, tempo, duration, key, mode, time_signature, artist_id, artist_name, release, song_hotttnesss, artist_hotttnesss, artist_familiarity, title, year).  Runs in ~30 seconds on the SSHFS mount.

### `extract_musics.py`

Walks the directory of per-song `.h5` files, extracts 100-dim aggregated segment features, similar-artist pairs, and artist-term pairs, writing them to batch Parquet files.

```bash
python extract_musics.py <data_root> <output_dir>
python extract_musics.py <data_root> <output_dir> -w 8   # 8 workers
```

Output per batch:
- `features_NNNN.parquet` -- N rows x 100 columns (pitch/timbre
  mean/std/min/max per song, plus loudness mean/std/min/max).
- `similar_NNNN.parquet` -- (track_id, similar_artist_id) pairs with
  empty strings filtered out.
- `terms_NNNN.parquet` -- (track_id, term) pairs.

Supports checkpoint/resume: a `checkpoint.json` file tracks processed paths line by line.  If the script is interrupted and re-run with the same output directory, already-processed files are skipped.

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

The dominant cost is **segment aggregation** (numpy mean/std/min/max on ~800x12 arrays) followed by **SSHFS I/O**.  Neither is improved by switching languages (numpy is already C-optimised), and the I/O path is gated by network latency to the remote server.

### Correctness

- All 10 000 output rows have **0 nulls, 0 NaN, 0 Inf**, and **0 zero-variance columns**.
- Three randomly selected files were cross-checked against their original HDF5 sources: `pitch_mean_0` values matched to 6 decimal places; similar-artist and term counts matched exactly.
- The checkpoint file contained exactly 10 000 lines, one per processed path.

### Full 1M Projection

Extrapolating from the 10K timing:

| Workers | Estimated wall-clock |
|--------:|:---------------------|
| 8       | ~9 hours             |

This is a one-time extraction.  With checkpoint/resume the job can be split across multiple sessions.
