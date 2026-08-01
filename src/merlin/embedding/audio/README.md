# MERLIN Audio Embeddings

This module trains and validates the PCA-based MERLIN audio encoder and builds a FAISS nearest-neighbor index over the resulting embeddings.

## Module layout

This directory is a Python package. Internal and downstream code should import through `merlin.embedding.audio`; the scripts also retain direct `spark-submit src/merlin/embedding/audio/<script>.py` entry points.

* `columns.py` defines the shared audio schema and feature order.
* `preprocess.py` fits preprocessing statistics and reapplies the frozen contract.
* `train_pca.py` fits the scaler/PCA encoder and publishes embeddings.
* `build_faiss.py` builds the audio nearest-neighbor index.
* `artifacts.py` owns encoder, manifest, lineage, and canonical-path contracts.
* `validate.py` validates encoder artifacts and runs the formal L1-1 experiment.
* `validate_faiss.py` validates index, mapping, manifest, and source alignment.
* `l1_stats.py` contains the statistical routines used by L1-1 validation.

## Canonical contract

The production C1 directory is `parquets_new/merlin/audio`. Formal C1 and L1-1 artifacts must fit and select exactly 128 PCA components, contain all one million tracks, and use the canonical filenames below. Candidate and Ranker stages consume this contract alongside C2's independent graph contract, using metadata and manifests rather than producer implementation details.

Limited or non-128-dimensional runs are experiments. They must use an isolated output directory and cannot be used for formal L1-1, the production FAISS loader, or Ranker training and evaluation.

## PCA dimension selection

Training supports two mutually exclusive selection modes:

* `--fixed-k K`: use exactly the first `K` principal components.
* `--target-variance T`: use the smallest number of principal components whose cumulative explained variance reaches `T`.

If neither option is supplied, training preserves the previous behavior and uses a fixed 128-dimensional embedding.

`--max-components K` controls how many PCA components are fitted before the final dimension is selected.

* In fixed-dimension mode, it defaults to `--fixed-k`.
* In target-variance mode, it defaults to `256`.
* `--fixed-k` cannot exceed `--max-components`.
* If the target variance is not reached, training fails and asks for a larger `--max-components`.

Each training invocation produces one encoder configuration. Use separate output directories for fixed-128, 90%, 95%, and smoke runs.

## Outputs

A successful training run publishes:

* `song_embeddings_audio.parquet`: `track_id` and the L2-normalized embedding.
* `audio_encoder_metadata.json`: selection mode, selected dimension, explained variance, feature order, preprocessing parameters, and lineage.
* `scaler_model`: fitted Spark `StandardScalerModel`.
* `pca_model`: fitted Spark `PCAModel`.
* `c1_manifest.json`: commit marker for the complete encoder artifact.

Training writes a sibling staging directory and publishes the output directory only after all files and Spark success markers are present.

## Train a fixed 128-dimensional encoder

Run from the repository root:

```bash
JAVA_TOOL_OPTIONS='-XX:UseAVX=0 -XX:UseSSE=2 -XX:-TieredCompilation' \
spark-submit --master 'local[6]' --driver-memory 5g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.shuffle.partitions=64 \
  src/merlin/embedding/audio/train_pca.py \
  --input parquets_new/prepared/song_audio_features_raw.parquet \
  --parent-manifest parquets_new/prepared/prepared_manifest.json \
  --output parquets_new/merlin/audio \
  --fixed-k 128 \
  --max-components 128 \
  --shuffle-partitions 64
```

Omitting both selection options also defaults to fixed 128 dimensions. The explicit arguments above make the formal contract visible in the command.

## Smoke test

Use a separate output directory:

```bash
JAVA_TOOL_OPTIONS='-XX:UseAVX=0 -XX:UseSSE=2 -XX:-TieredCompilation' \
spark-submit --master 'local[6]' --driver-memory 5g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.shuffle.partitions=64 \
  src/merlin/embedding/audio/train_pca.py \
  --input parquets_new/prepared/song_audio_features_raw.parquet \
  --parent-manifest parquets_new/prepared/prepared_manifest.json \
  --output parquets_new/merlin/audio-smoke \
  --fixed-k 128 \
  --limit 10000 \
  --shuffle-partitions 64
```

This checks the pipeline on a smaller input. It does not establish the final explained-variance or retrieval-quality result for the full dataset.

Because the encoder already contains only 10,000 rows, build its complete smoke index without passing another `--limit`:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  src/merlin/embedding/audio/build_faiss.py \
  --input parquets_new/merlin/audio-smoke/song_embeddings_audio.parquet \
  --output parquets_new/merlin/audio-smoke \
  --shuffle-partitions 64
```

For a partial-index smoke run, all three FAISS artifact names must be separate from the production names. This prevents a limited or non-128-dimensional test index from replacing the formal index or its manifest:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  src/merlin/embedding/audio/build_faiss.py \
  --input parquets_new/merlin/audio-smoke/song_embeddings_audio.parquet \
  --output parquets_new/merlin/audio-smoke \
  --limit 1000 \
  --index-name index_audio_smoke_partial.faiss \
  --track-ids-name index_audio_smoke_partial_track_ids.parquet \
  --manifest-name index_audio_smoke_partial_manifest.json \
  --shuffle-partitions 64
```

## Explained-variance experiments

### 90% cumulative explained variance

```bash
JAVA_TOOL_OPTIONS='-XX:UseAVX=0 -XX:UseSSE=2 -XX:-TieredCompilation' \
spark-submit --master 'local[6]' --driver-memory 5g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.shuffle.partitions=64 \
  src/merlin/embedding/audio/train_pca.py \
  --input parquets_new/prepared/song_audio_features_raw.parquet \
  --parent-manifest parquets_new/prepared/prepared_manifest.json \
  --output parquets_new/merlin/audio-var90 \
  --target-variance 0.90 \
  --max-components 256 \
  --shuffle-partitions 64
```

### 95% cumulative explained variance

```bash
JAVA_TOOL_OPTIONS='-XX:UseAVX=0 -XX:UseSSE=2 -XX:-TieredCompilation' \
spark-submit --master 'local[6]' --driver-memory 5g \
  --conf spark.ui.enabled=false \
  --conf spark.sql.shuffle.partitions=64 \
  src/merlin/embedding/audio/train_pca.py \
  --input parquets_new/prepared/song_audio_features_raw.parquet \
  --parent-manifest parquets_new/prepared/prepared_manifest.json \
  --output parquets_new/merlin/audio-var95 \
  --target-variance 0.95 \
  --max-components 256 \
  --shuffle-partitions 64
```

If 256 components do not reach the requested target, increase the limit, for example:

```bash
--max-components 384
```

Inspect these fields in `audio_encoder_metadata.json`:

```json
{
  "selection_mode": "target_variance",
  "target_variance": 0.9,
  "fixed_k": null,
  "requested_max_components": 256,
  "max_components": 256,
  "selected_k": 91,
  "selected_cumulative_explained_variance": 0.9003,
  "target_variance_reached": true
}
```

The numbers above are illustrative.

## Validate an encoder

Validate the canonical encoder artifact:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  src/merlin/embedding/audio/validate.py \
  --mode artifact \
  --output parquets_new/merlin/audio \
  --expected-rows 1000000 \
  --shuffle-partitions 64
```

For an isolated dimension-selection experiment, opt in explicitly:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  src/merlin/embedding/audio/validate.py \
  --mode artifact \
  --output parquets_new/merlin/audio-var90 \
  --expected-rows 1000000 \
  --allow-noncanonical-dimension \
  --shuffle-partitions 64
```

The validator must read the expected embedding dimension from `audio_encoder_metadata.json["selected_k"]`. It must not assume dimension 128. The explicit flag is required because only fitted-and-selected PCA-128 artifacts are eligible for the canonical C1 path and formal L1-1 evidence.

It should check:

* Output files and Spark success markers exist.
* Row count and distinct `track_id` count are correct.
* Every embedding has length `selected_k`.
* Embeddings contain no null, NaN, or infinite values.
* Embeddings are L2-normalized.
* The fitted PCA model dimension equals `max_components`.
* The published embedding dimension equals `selected_k`.

## Build FAISS

Build the canonical production index from the formal encoder:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  src/merlin/embedding/audio/build_faiss.py \
  --input parquets_new/merlin/audio/song_embeddings_audio.parquet \
  --output parquets_new/merlin/audio \
  --shuffle-partitions 64
```

`build_faiss.py` reads the vector dimension from `audio_encoder_metadata.json["selected_k"]`.

It writes:

* `index_audio.faiss`: FAISS `IndexFlatIP` index.
* `index_audio_track_ids.parquet`: `row_id` to `track_id` mapping.
* `index_audio_manifest.json`: dimension, row count, hashes, encoder run ID, and partial-index status.

The default names are reserved for a complete production build. When using `--limit` or custom index names in `parquets_new/merlin/audio`, pass non-default names for the index, mapping, and manifest together. A partial index is not discoverable by the standard C2 loader when it uses a custom manifest name.

Inner product equals cosine similarity because embeddings are L2-normalized.

### Partial FAISS index

To build a limited index, use isolated output names, for example:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  src/merlin/embedding/audio/build_faiss.py \
  --input parquets_new/merlin/audio-smoke/song_embeddings_audio.parquet \
  --output parquets_new/merlin/audio-smoke \
  --limit 10000 \
  --index-name index_audio_partial.faiss \
  --track-ids-name index_audio_partial_track_ids.parquet \
  --manifest-name index_audio_partial_manifest.json
```

This indexes the first 10,000 embeddings after sorting by `track_id`. When a limited build targets the production directory, all three names must differ from their canonical counterparts. The manifest records:

```json
{
  "partial_index": true,
  "requested_limit": 10000
}
```

Do not treat a partial index as a production artifact.

## Validate FAISS

Canonical validation uses the default names and requires PCA-128:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  src/merlin/embedding/audio/validate_faiss.py \
  --output parquets_new/merlin/audio \
  --expected-rows 1000000 \
  --shuffle-partitions 64
```

An isolated noncanonical experiment must opt in explicitly:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  src/merlin/embedding/audio/validate_faiss.py \
  --embeddings parquets_new/merlin/audio-var90/song_embeddings_audio.parquet \
  --output parquets_new/merlin/audio-var90 \
  --expected-rows 1000000 \
  --allow-noncanonical-dimension \
  --shuffle-partitions 64
```

The validator should verify:

* The index and mapping exist.
* Index size, mapping size, and mapping uniqueness agree.
* `index.d` equals the FAISS manifest dimension.
* The FAISS dimension equals encoder metadata `selected_k`.
* Sample top-K searches retrieve the query track itself.

The validator must not hard-code dimension 128.

## Validate L1-1 feature sanity

Compare cleaned and scaled pre-PCA cosine similarity with the canonical PCA-128 cosine similarity on the same same-artist, same-release, and matched-random pairs:

```bash
spark-submit --master 'local[6]' --driver-memory 5g \
  src/merlin/embedding/audio/validate.py \
  --mode l1 \
  --raw-input parquets_new/prepared/song_audio_features_raw.parquet \
  --songs-metadata parquets_new/prepared/songs_metadata.parquet \
  --output parquets_new/merlin/audio \
  --pair-count 10000 \
  --bootstrap-samples 2000 \
  --seed 42 \
  --shuffle-partitions 64
```

Use `--mode all` to run artifact integrity checks first and L1-1 in the same Spark session. The default `--mode artifact` preserves the fast standalone encoder-validation workflow.

The validator must use the saved preprocessing parameters, scaler, PCA model, and `selected_k`. It must not fit another model. Non-128 experiment artifacts can run `--mode artifact`, but cannot run formal `l1` or `all` validation because the frozen L1-1 contract compares pre-PCA with PCA-128.

The same frozen preprocessing contract is reused by the Ranker when it reconstructs cleaned pre-PCA vectors for Set-B tune/confirm and Set-C development groups. Changing C1 feature order, medians, clipping, time-signature encoding, scaler statistics, or dimension metadata therefore requires the corresponding manifest contract to change; downstream code must never silently refit these values.

`validation_report.json` should include:

* `selection_mode`, `target_variance`, and `selected_k`.
* Similarity distributions.
* Hedges' g against matched-random pairs.
* Bootstrap 95% confidence intervals.
* Pre-/post-PCA preservation diagnostics.
* Exact embedding-reproduction error.

`--allow-partial-pairs` is for smoke artifacts only. Such a run is marked `SMOKE_PASS` and cannot support the formal L1-1 conclusion.

## Recommended layout

```text
parquets_new/merlin/
├── audio/                 # canonical full PCA-128 encoder and FAISS index
├── audio-fixed128/        # optional comparison run
├── audio-var90/
├── audio-var95/
└── audio-smoke/
```

Do not reuse an output directory for different configurations. Training publishes the encoder as a complete directory and replaces an existing artifact at that path.

Compare at least:

* Selected dimension.
* Cumulative explained variance.
* Same-artist and same-release similarities.
* Separation from matched-random pairs.
* Hedges' g and confidence intervals.
* Retrieval recall or top-K neighbor preservation.
* FAISS index size and query latency.

The MERLIN Candidate and Ranker pipeline discovers only `parquets_new/merlin/audio`. The comparison directories are isolated experiments and are not runtime candidates.
