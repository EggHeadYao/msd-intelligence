# Year Prediction Dataset Contract

The dataset layer freezes valid year labels and an artist-disjoint train, validation, and test assignment. It does not contain audio predictors.

## Inputs

- `raw/songs_scalar.parquet`: one row per MSD track; missing years are Arrow nulls.
- `raw/audio_features/`: the `shared_audio_628_v1` feature batches used for coverage validation.
- `source/artists_train.txt` and `source/artists_test.txt`: official MSD artist lists pinned by checksum.

## Build

The builder refuses to overwrite an existing output directory. Use a staging directory when replacing a frozen contract:

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  src/year_prediction/src/data/build_dataset.py \
  --output parquets/year_prediction/dataset.staging
```

## Validate

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  src/year_prediction/src/data/validate_dataset.py \
  --dataset parquets/year_prediction/dataset.staging \
  --reference-split parquets/year_prediction/dataset/split_assignments.parquet
```

## Outputs

- `labelled_tracks.parquet`: `track_id`, `artist_id`, `year`, and `split` for every valid year label.
- `split_assignments.parquet`: the same track and artist assignment without the label, for independent split reuse and auditing.
- `manifest.json`: input checksums, schemas, split rules, counts, and the artist-assignment checksum.
