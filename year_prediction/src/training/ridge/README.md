# T90 Ridge Training

This directory prepares the T90 feature view, trains the custom Spark SGD Ridge model, and validates its data and model artifacts.

## Prepare T90 Data

`prepare_t90.py` reads the labeled rows in `features/t90.parquet`. It fits missing-value means on the train split, fits sample standard deviations on the mean-imputed train split, and applies those statistics to train, validation, and test without refitting.

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/training/ridge/prepare_t90.py \
  --input parquets/year_prediction/features/t90.parquet \
  --feature-manifest parquets/year_prediction/features/manifest.json \
  --output parquets/year_prediction/training/t90 \
  --shuffle-partitions 32 \
  --output-partitions 32
```

The builder refuses to overwrite an existing output directory.

## Validate T90 Data

`validate_t90_data.py` independently recomputes train statistics and every transformed vector from the source T90 table. It also checks schemas, split counts, artist isolation, finite values, target normalization, feature dimension, and standardized train moments.

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/training/ridge/validate_t90_data.py \
  --input parquets/year_prediction/training/t90 \
  --shuffle-partitions 32
```

## T90 Data Outputs

- `vectors.parquet`: identifiers, year, normalized year, a fixed 90-value `features` array, and split.
- `manifest.json`: source fingerprint, split counts, target contract, exact feature order, train-only statistics, missing counts, and output schema.

The Ridge trainer consumes only this validated artifact.
