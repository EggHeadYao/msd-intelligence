# Year Prediction Feature Views

This layer projects the frozen shared audio and dataset contracts into two model-facing tables. It does not fit imputation, scaling, low-variance filtering, or model parameters.

## Inputs

- `raw/audio_features/`: `track_id` plus the 628 predictors in `shared_audio_628_v1`.
- `raw/songs_scalar.parquet`: global audio fields, track identity, artist identity, and nullable year.
- `dataset/labelled_tracks.parquet`: frozen artist-disjoint split for valid year labels.

## Build

```bash
spark-submit --master 'local[1]' --driver-memory 3g \
  p1team02/year_prediction/src/features/build_features.py \
  --output parquets/year_prediction/features \
  --shuffle-partitions 32
```

The builder refuses to overwrite an existing output directory.

## Validate

```bash
spark-submit --master 'local[2]' --driver-memory 4g \
  p1team02/year_prediction/src/features/validate_features.py \
  --features parquets/year_prediction/features \
  --shuffle-partitions 32
```

## Outputs

- `t90.parquet`: four audit columns followed by the official 90-dimensional timbre block.
- `full_tabular.parquet`: four audit columns followed by 580 shared predictors and 14 global predictors.
- `manifest.json`: source checksums, exact schemas and column orders, feature groups, formulas, units, missing-value rules, and row counts.

Both tables retain all 1,000,000 tracks. The 515,576 supervised rows have `year` and `split`; both fields are null for unlabeled tracks. Audit columns never enter a model.
