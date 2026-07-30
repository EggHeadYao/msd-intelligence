# Metadata Feature Views

This pipeline adds metadata predictors without changing the existing audio-only contract. All target statistics use train artists only. A training artist is removed from its own tag statistics, similarity self-edges are discarded, and validation and test years are never read as predictors.

## Outputs

- `metadata_only.parquet`: scalar metadata, location, tag indicators, tag-year priors, and similarity-ranked artist-year summaries.
- `audio_metadata.parquet`: the 594 audio predictors joined with 72 compact metadata summaries and 96 high-frequency tag indicators. The summaries include the first 20 train-neighbor years and cumulative top-1/3/5/10/20 statistics. Limiting the fused view to 64 terms and 32 MusicBrainz tags keeps training within a bounded memory footprint.
- `manifest.json`: ordered predictor contracts, selected tags, smoothing state, row counts, and source paths.

## Run

```bash
src/year_prediction/.synapseml-venv/bin/python \
  src/year_prediction/src/features/metadata/export.py \
  --input msd/AdditionalFiles \
  --output parquets/year_prediction/raw/metadata

src/year_prediction/.synapseml-venv/bin/spark-submit \
  --driver-memory 4g \
  src/year_prediction/src/features/metadata/build.py \
  --metadata parquets/year_prediction/raw/metadata \
  --output parquets/year_prediction/features/metadata \
  --shuffle-partitions 32

src/year_prediction/.synapseml-venv/bin/spark-submit \
  --driver-memory 4g \
  src/year_prediction/src/features/metadata/validate.py \
  --input parquets/year_prediction/features/metadata
```
