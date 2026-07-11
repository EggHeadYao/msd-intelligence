# MERLIN Audio Embeddings

This module trains and validates the PCA-based audio encoder.

## Train

Run from the repository root:

```bash
spark-submit p1team02/merlin/embedding/audio/train_pca.py \
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
spark-submit p1team02/merlin/embedding/audio/validate.py \
  --output parquets/merlin/audio \
  --expected-rows 1000000 \
  --shuffle-partitions 64
```

- Checks that the embedding table, metadata file, scaler model, and PCA model exist.
- Checks row count and distinct `track_id` count.
- Checks that every embedding has the selected dimension.
- Checks that embeddings contain no null, NaN, or infinite values.
- Checks that embeddings are L2-normalized.
