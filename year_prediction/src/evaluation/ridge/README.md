# Ridge Evaluation

- `evaluate.py`: predicts every T90 test track with a saved Ridge model and writes a fixed quality report.
- `validate.py`: reloads a saved Ridge model, recomputes train and validation results, and verifies every saved validation prediction.

`validate.py` checks artifact consistency. It does not retrain the model, select hyperparameters, or evaluate the test benchmark.

## Validate a saved model:

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  year_prediction/src/evaluation/ridge/validate.py \
  --model parquets/year_prediction/models/<model_id>
```

## Evaluate the test benchmark:

```bash
spark-submit --master 'local[4]' --driver-memory 3g \
  year_prediction/src/evaluation/ridge/evaluate.py \
  --model parquets/year_prediction/models/<model_id> \
  --prediction-partitions 8
```

The default output is `parquets/year_prediction/results/experiment_a/ridge/<model_id>/test/`.

- `predictions.parquet`: track and artist IDs, target year, normalized prediction, raw and clipped prediction years, and absolute error.
- `metrics.json`: overall MAE, RMSE, median absolute error, within-five/ten-year rates, signed error, raw metrics, and macro-decade MAE.
- `metrics_by_decade.json`: count, MAE, RMSE, and signed error for each target decade.
- `run_metadata.json`: model and feature checksums, test counts, Spark runtime, and evaluation timings.
