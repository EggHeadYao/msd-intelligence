# MERLIN Audio Embeddings

This module trains and validates the PCA-based audio encoder.

## Train

Run from the repository root:

```bash
spark-submit --driver-memory 4g p1team02/merlin/embedding/audio/train_pca.py \
  --input parquets/prepared/song_audio_features_raw.parquet \
  --output parquets/merlin/audio \
  --target-variance 0.95 \
  --shuffle-partitions 64
```

- `song_embeddings_audio.parquet`: Stores `track_id` and the normalized audio embedding.
- `audio_encoder_metadata.json`: Stores feature columns, selected dimension, explained variance, and preprocessing metadata.
- `scaler_model`: Stores the fitted Spark `StandardScalerModel`.
- `pca_model`: Stores the fitted Spark `PCAModel`.

## Validate

Validate the full output:

```bash
spark-submit --driver-memory 4g p1team02/merlin/embedding/audio/validate.py \
  --output parquets/merlin/audio \
  --expected-rows 1000000 \
  --shuffle-partitions 64
```

- Checks that the embedding table, metadata file, scaler model, and PCA model exist.
- Checks row count and distinct `track_id` count.
- Checks that every embedding has the selected dimension.
- Checks that embeddings contain no null, NaN, or infinite values.
- Checks that embeddings are L2-normalized.

## Build FAISS

Build the audio nearest-neighbor index:

```bash
spark-submit --driver-memory 4g p1team02/merlin/embedding/audio/build_faiss.py \
  --input parquets/merlin/audio/song_embeddings_audio.parquet \
  --output parquets/merlin/audio \
  --shuffle-partitions 64
```

- `index_audio.faiss`: Stores the FAISS inner-product index over normalized audio embeddings.
- `index_audio_track_ids.parquet`: Stores `row_id` and `track_id`; `row_id` matches the FAISS vector order.

## Validate FAISS

Validate the saved audio index:

```bash
spark-submit --driver-memory 4g p1team02/merlin/embedding/audio/validate_faiss.py \
  --embeddings parquets/merlin/audio/song_embeddings_audio.parquet \
  --output parquets/merlin/audio \
  --expected-rows 1000000 \
  --shuffle-partitions 64
```

- Checks that the FAISS index and track-id mapping exist.
- Checks index size, embedding dimension, mapping size, and mapping uniqueness.
- Runs sample top-K searches and verifies that query tracks retrieve themselves.
- Uses inner product because embeddings are L2-normalized, so scores equal cosine similarity.
