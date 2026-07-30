# MERLIN Prepare

Build the canonical tables consumed by MERLIN audio preprocessing and graph indexing. Preparation is deterministic ETL: it does not train a model or create embeddings.

## Inputs

The command reads one input namespace, normally `parquets_new`, containing:

- `songs_scalar.parquet`: the 23-column Summary HDF5 export.
- `musics/feature_contract.json`: the frozen `shared_audio_628_v1` version, feature count, ordered columns, and order hash.
- `musics/features_*.parquet`: `track_id` plus the ordered 628 nullable `float64` audio features.
- `track_metadata.parquet`
- `artist_term.parquet`
- `artist_similarity_edges.parquet`

The extractor contract version, 629-column order/hash, input schemas, key uniqueness, non-empty relationship keys, and cross-source track/song/artist identity are checked before output is initialized. Per-track `terms_*.parquet` and `similar_*.parquet` files are never read.

## Outputs

The output directory contains exactly three Parquet datasets:

- `songs_metadata.parquet`: one row per extracted track with stable track/song/artist IDs, display metadata, MusicBrainz artist ID, real `release_7digitalid`, `track_7digitalid`, year availability, and audit-only popularity fields. Release names are display text and are never used as entity keys.
- `song_audio_features_raw.parquet`: one row per track containing the frozen MERLIN projection of 552 shared array features plus 11 Summary analysis values, for a 563-dimensional raw view (`track_id` is the additional key column). It excludes the 50 raw half-pooling columns, 24 key-relative half-pooling columns, two half masks, `danceability`, `energy`, year, and popularity. Undefined array-derived values and unknown raw tempo/time signature remain null for C1 preprocessing to mask and impute; NaN and Inf are rejected.
- `graph_edges.parquet`: an unweighted typed graph partitioned by `edge_type`.

The graph schema is:

```text
src_type, src_id, dst_type, dst_id, directed, edge_type
```

It contains only these canonical relations:

| Edge type           | Stored direction                                            | Full-catalog rows |
| ------------------- | ----------------------------------------------------------- | ----------------: |
| `track_artist`      | track to artist, traversed both ways                        |         1,000,000 |
| `track_release`     | track to positive `release_7digitalid`, traversed both ways |           999,997 |
| `artist_term`       | artist to term, traversed both ways                         |         1,109,381 |
| `artist_similarity` | artist to similar artist, directed                          |         2,201,916 |
| Total               |                                                             |         5,311,294 |

There is no graph `weight` column. Uniform transitions need no stored weight; the P3 capped-IDF sampling weights are derived later while building C2 adjacency.

## Safety and lineage

An existing output is rejected by default. `--reset-output` may remove only a directory carrying a matching MERLIN ownership marker and matching protected input paths. Preparation writes `prepared_manifest.json` with code state, configuration, input row counts, schema hashes, output columns, and lineage paths. Its status remains `initialized` until `validate.py` passes and marks it `valid`.

Downstream C1/C2 jobs must consume only a `valid` prepared manifest.

## Commands

From the repository root with the project environment activated, expose the Python packages under `src/`:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m merlin.prepare.prepare \
  --input parquets_new \
  --output <output_dir> \
  --shuffle-partitions 64
```

Local execution defaults to four Spark worker threads and a 4 GiB JVM heap so the 629-column extracted audio batches do not exhaust memory through excessive concurrent Parquet reads. Machines with different resources may override these settings with `--spark-master` and `--driver-memory`.

Validate the full-catalog output:

```bash
python3 -m merlin.prepare.validate \
  --prepared <output_dir>  \
  --shuffle-partitions 64
```

The validator checks the exact three-table layout and ordered schemas; one-to-one track coverage; real release-ID coverage; null/finite contracts; binary masks; four exact edge types and counts; typed endpoint semantics; direction flags; duplicate pairs; metadata-to-graph mappings; and manifest schema lineage. Any failed check exits nonzero and leaves the artifact unavailable to downstream stages.
