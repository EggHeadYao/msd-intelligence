# Spark Ordinal-MoE year prediction

Ordinal-MoE is trained with distributed gradient descent. Spark reads and
partitions the 594-feature contract, computes train-only standardization,
calculates analytic gradients inside each partition, and combines them with a
tree reduction. The driver applies Adam to the reduced gradient. NumPy is used
only as the numerical kernel inside Spark tasks; there is no pandas, PyTorch, or
single-machine training path.

The model has four connected outputs:

- a CORAL-style ordinal head with 89 ordered year thresholds;
- a softmax decade gate;
- ten bounded linear year experts;
- a robust direct-year head used as an auxiliary objective.

Huber losses reduce the influence of large year residuals. A consistency term
keeps the ordinal, expert, and direct estimates compatible. Validation MAE drives
early stopping, while test data is not read during normal training.

## Training

```bash
spark-submit \
  --master spark://spark-master:7077 \
  year_prediction/src/training/ordinal_moe/ordinal_moe_train.py \
  --input data/year_prediction/full_tabular.parquet \
  --manifest data/year_prediction/manifest.json \
  --output artifacts/ordinal-moe \
  --partitions 64
```

## Frozen test evaluation

```bash
spark-submit \
  --master spark://spark-master:7077 \
  year_prediction/src/evaluation/ordinal_moe/evaluate.py \
  --model-root artifacts/ordinal-moe \
  --input data/year_prediction/full_tabular.parquet \
  --manifest data/year_prediction/manifest.json \
  --output artifacts/ordinal-moe-test
```

## Artist-isolated OOF stacking

First build deterministic artist folds. Then run each model once per held-out
fold and merge the fold predictions. Ordinal-MoE uses `--fold-assignments` and
`--validation-fold`; LightGBM uses `lightgbm_oof.py --fold`. Fit stack weights
only from the merged OOF predictions.

```bash
spark-submit year_prediction/src/training/ordinal_moe/build_artist_folds.py \
  --input data/year_prediction/full_tabular.parquet \
  --manifest data/year_prediction/manifest.json \
  --output artifacts/folds

spark-submit year_prediction/src/training/ordinal_moe/stack_predictions.py \
  --ordinal artifacts/ordinal-oof/predictions.parquet \
  --lightgbm artifacts/lightgbm-oof/predictions.parquet \
  --output artifacts/stack
```

Never fit preprocessing, early stopping, or stack weights on the fixed test
split.
