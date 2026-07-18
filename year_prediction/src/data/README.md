# Year Prediction Dataset Contract

This module builds a leakage-safe supervised dataset from the prepared MSD metadata and audio-feature Parquet tables. It preserves the official test artist set and deterministically divides every other labeled artist between training and validation.

## Inputs

- `parquets/prepared/songs_metadata.parquet`
- `parquets/prepared/song_audio_features_raw.parquet`
- `parquets/year_prediction/source/artists_train.txt`
- `parquets/year_prediction/source/artists_test.txt`

The artist files are pinned to MSD commit `0c276e289606d5bd6f3991f713e7e9b1d4384e44`. The official files contain 25,398 train artists and 2,822 test artists. They omit three of the 28,223 labeled artists, so the builder preserves the official test set and includes those three artists in the deterministic train/validation hash split with every other non-test artist.

```bash
mkdir -p parquets/year_prediction/source

curl -L \
  -o parquets/year_prediction/source/artists_train.txt \
  https://raw.githubusercontent.com/tbertinmahieux/MSongsDB/0c276e289606d5bd6f3991f713e7e9b1d4384e44/Tasks_Demos/YearPrediction/artists_train.txt

curl -L \
  -o parquets/year_prediction/source/artists_test.txt \
  https://raw.githubusercontent.com/tbertinmahieux/MSongsDB/0c276e289606d5bd6f3991f713e7e9b1d4384e44/Tasks_Demos/YearPrediction/artists_test.txt
```

## Build

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/data/build_dataset.py
```

## Validate

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/data/validate_dataset.py
```

## Outputs

- `supervised_features.parquet`: `track_id`, `artist_id`, `year`, 109 raw audio predictor columns, and the partition column `split`.
- `split_assignments.parquet`: `track_id`, `artist_id`, and `split`, used to audit artist isolation and reproduce the data split.
- `manifest.json`: source paths and fingerprints, schema types, year rules, build parameters, row counts, split statistics, omitted official artists, and the artist-assignment checksum.
