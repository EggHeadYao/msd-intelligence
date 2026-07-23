# MERLIN Audio Embeddings

This module trains and validates the PCA-based MERLIN audio encoder and builds a
FAISS nearest-neighbor index over the resulting embeddings.

## PCA dimension selection

Training supports two mutually exclusive selection modes:

* `--fixed-k K`: use exactly the first `K` principal components.
* `--target-variance T`: use the smallest number of principal components whose
  cumulative explained variance reaches `T`.

If neither option is supplied, training preserves the previous behavior and
uses a fixed 128-dimensional embedding.

`--max-components K` controls how many PCA components are fitted before the
final dimension is selected.

* In fixed-dimension mode, it defaults to `--fixed-k`.
* In target-variance mode, it defaults to `256`.
* `--fixed-k` cannot exceed `--max-components`.
* If the target variance is not reached, training fails and asks for a larger
  `--max-components`.

Each training invocation produces one encoder configuration. Use separate
output directories for fixed-128, 90%, 95%, and smoke runs.

## Outputs

A successful training run publishes:

* `song_embeddings_audio.parquet`: `track_id` and the L2-normalized embedding.
* `audio_encoder_metadata.json`: selection mode, selected dimension, explained
  variance, feature order, preprocessing parameters, and lineage.
* `scaler_model`: fitted Spark `StandardScalerModel`.
* `pca_model`: fitted Spark `PCAModel`.
* `c1_manifest.json`: commit marker for the complete encoder artifact.

Training writes a sibling staging directory and publishes the output directory
only after all files and Spark success markers are present.

## Train a fixed 128-dimensional encoder

Run from the repository root:

```bash
JAVA_TOOL_OPTIONS='-XX:UseAVX=0 -XX:UseSSE=2 -XX:-TieredCompilation' \
spark-submit --master 'local[6]' --driver-memory 5g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.shuffle.partitions=64 \
  merlin/embedding/audio/train_pca.py \
  --input parquets_new/prepared/song_audio_features_raw.parquet \
  --parent-manifest parquets_new/prepared/prepared_manifest.json \
  --output parquets_new/merlin/audio-fixed128 \
  --fixed-k 128 \
  --shuffle-partitions 64
```

## Smoke test

Use a separate output directory:

```bash
JAVA_TOOL_OPTIONS='-XX:UseAVX=0 -XX:UseSSE=2 -XX:-TieredCompilation' \
spark-submit --master 'local[6]' --driver-memory 5g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.shuffle.partitions=64 \
  merlin/embedding/audio/train_pca.py \
  --input parquets_new/prepared/song_audio_features_raw.parquet \
  --parent-manifest parquets_new/prepared/prepared_manifest.json \
  --output parquets_new/merlin/audio-smoke \
  --fixed-k 128 \
  --limit 10000 \
  --shuffle-partitions 64
```

This checks the pipeline on a smaller input. It does not establish the final
explained-variance or retrieval-quality result for the full dataset.

Validate the full output:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  merlin/embedding/audio/validate.py \
  --output parquets_new/merlin/audio \
  --expected-rows 1000000 \
  --shuffle-partitions 64
```

- Checks that the embedding table, metadata file, scaler model, and PCA model exist.
- Checks row count and distinct `track_id` count.
- Checks that every embedding has the selected dimension.
- Checks that embeddings contain no null, NaN, or infinite values.
- Checks that embeddings are L2-normalized.
- Loads the saved scaler and PCA models and compares them with encoder metadata.
- Validates all embedding properties in one aggregate scan.

## Build FAISS

Build the audio nearest-neighbor index:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  merlin/embedding/audio/build_faiss.py \
  --input parquets_new/merlin/audio/song_embeddings_audio.parquet \
  --output parquets_new/merlin/audio \
  --shuffle-partitions 64
```

- `index_audio.faiss`: Stores the FAISS inner-product index over normalized audio embeddings.
- `index_audio_track_ids.parquet`: Stores `row_id` and `track_id`; `row_id` matches the FAISS vector order.

It writes:

* `index_audio.faiss`: FAISS `IndexFlatIP` index.
* `index_audio_track_ids.parquet`: `row_id` to `track_id` mapping.
* `index_audio_manifest.json`: dimension, row count, hashes, encoder run ID, and
  partial-index status.

The default names are reserved for a complete production build. When using
`--limit` or custom index names in `parquets_new/merlin/audio`, pass non-default
names for the index, mapping, and manifest together. A partial index is not
discoverable by the standard C2 loader when it uses a custom manifest name.

Inner product equals cosine similarity because embeddings are L2-normalized.

### Partial FAISS index

Using (with isolated output names):

```bash
--limit 10000
```

The command must also provide non-default `--index-name`, `--track-ids-name`,
and `--manifest-name` values when writing into the production audio directory.

builds an index over only the first 10,000 embeddings after sorting by
`track_id`. The manifest records:

```json
{
  "partial_index": true,
  "requested_limit": 10000
}
```

Do not treat a partial index as a production artifact.

## Validate FAISS

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  merlin/embedding/audio/validate_faiss.py \
  --embeddings parquets_new/merlin/audio/song_embeddings_audio.parquet \
  --output parquets_new/merlin/audio \
  --expected-rows 1000000 \
  --shuffle-partitions 64
```

- Checks that the FAISS index and track-id mapping exist.
- Checks index size, embedding dimension, mapping size, and mapping uniqueness.
- Runs sample top-K searches and verifies that query tracks retrieve themselves.
- Uses inner product because embeddings are L2-normalized, so scores equal cosine similarity.

## Validate L1-1 Feature Sanity

Compare cleaned and scaled pre-PCA cosine with final PCA-128 cosine on the exact
same same-artist, same-release, and matched-random pairs:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  merlin/embedding/audio/validate_feature_sanity.py \
  --raw-input parquets_new/prepared/song_audio_features_raw.parquet \
  --songs-metadata parquets_new/prepared/songs_metadata.parquet \
  --output parquets_new/merlin/audio \
  --pair-count 10000 \
  --bootstrap-samples 2000 \
  --seed 42 \
  --shuffle-partitions 64
```

The validator applies only the medians, clipping bounds, feature order, scaler,
and PCA model saved during training. It does not fit a second model. It writes
`validation_report.json` with similarity distributions, Hedges' g versus random,
bootstrap 95% confidence intervals, pairwise pre/post-PCA diagnostics, and exact
embedding-reproduction error.

`--allow-partial-pairs` is for small smoke artifacts only. Such a run is marked
`SMOKE_PASS` and cannot support the formal L1-1 conclusion.
