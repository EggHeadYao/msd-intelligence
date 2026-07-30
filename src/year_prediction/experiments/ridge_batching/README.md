# Ridge Batch-Fraction Comparison

This experiment compares full-batch and reproducibly sampled mini-batch updates for the T90 Ridge baseline. Every run uses the same data, initialization, learning rate, L2 penalty, 100 parameter updates, `local[4]`, and 3 GB driver memory. Only `batch_fraction` changes.

`summary.csv` records actual sampled rows, equivalent full-data passes, training time, and independently recomputed validation and test quality. Timings are single local runs and should be interpreted as an engineering comparison rather than a statistically precise benchmark.

## Comparison Files

- `summary.csv`: headline work, timing, validation, and test measurements.
- `efficiency.csv`: batch sizes, data passes, gradient time, speedup, and time reduction.
- `quality.csv`: train, validation, and test metrics under the common quality contract.
- `convergence.csv`: validation checkpoints against iterations, time, and data passes.
- `time_to_quality.csv`: first checkpoint reaching two common validation-MAE targets.
- `results/<model-id>/`: original training history, metrics, runtime metadata, and test reports.

Large model weights and prediction Parquet files are intentionally excluded.

## Generate Figures

```bash
python3 src/year_prediction/experiments/ridge_batching/scripts/plot_convergence.py
python3 src/year_prediction/experiments/ridge_batching/scripts/plot_tradeoff.py
```

The scripts write presentation-ready PDF and PNG versions of `ridge_batching_convergence` and `ridge_batching_tradeoff` to `src/slides/img/`.

## Findings

- The 25% and 10% runs reduce total training time by about 23% while preserving test quality.
- Test MAE spans only 0.0045 years and test RMSE spans only 0.0002 years across all runs.
- Both mini-batch runs reach the full-batch final validation MAE near iteration 95, using about 24 and 9.5 equivalent data passes respectively.
- The 10% run performs much less gradient work than the 25% run but has nearly identical wall-clock time. Spark data traversal and job overhead therefore limit the local speedup.
- The 10% convergence curve fluctuates more near the optimum, as expected from its noisier gradient estimates.
