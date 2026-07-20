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

## Train Ridge

`train.py` loads only the train and validation partitions, initializes zero weights and the intercept to the train-label mean, and performs full-batch Spark SGD with the configuration in `config/experiment_a/ridge_t90.json`. The held-out test partition is not read during training or model selection.

```bash
spark-submit --master 'local[4]' --driver-memory 3g \
  p1team02/year_prediction/src/training/ridge/train.py \
  --config p1team02/year_prediction/config/experiment_a/ridge_t90.json
```

Use a new `model_id` for every immutable run. The trainer refuses to overwrite an existing model directory unless `--overwrite` is explicitly supplied.

## Validate Ridge

`evaluation/ridge/validate.py` independently reloads the saved model and T90 contract, recomputes the final objective, gradient norm, train and validation metrics, and verifies every saved validation prediction.

```bash
spark-submit --master 'local[4]' --driver-memory 3g \
  p1team02/year_prediction/src/evaluation/ridge/validate.py \
  --model parquets/year_prediction/models/<model_id>
```

## Ridge Files

- `data.py`: validates the T90 manifest and loads only artist-disjoint train and validation rows.
- `objectives.py`: defines linear prediction, Ridge objective partials, and gradient aggregation.
- `optimizer.py`: applies one gradient-descent parameter update and computes gradient norms.
- `distributed.py`: evaluates metrics, predictions, and direct full-batch Spark reductions.
- `train.py`: runs the configured optimization loop and writes an immutable model bundle.

## Model Outputs

- `model.json`: weights, intercept, regularization, target contract, and feature-manifest identity.
- `history.json`: per-iteration objective, gradient norm, validation MAE, and timing.
- `metrics.json`: final train and validation quality plus convergence information.
- `run_metadata.json`: exact configuration, row counts, Spark runtime, checksums, and stage timings.
- `validation_predictions.parquet`: one independently verifiable prediction for each validation track.
