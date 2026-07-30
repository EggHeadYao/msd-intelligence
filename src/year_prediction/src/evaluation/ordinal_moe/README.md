# Spark Ordinal-MoE evaluation

Run frozen test-only evaluation with `spark-submit`:

```bash
spark-submit \
  src/year_prediction/src/evaluation/ordinal_moe/evaluate.py \
  --model-root artifacts/ordinal-moe \
  --input data/full_tabular.parquet \
  --manifest data/manifest.json \
  --output artifacts/ordinal-moe-test
```

The evaluator loads saved train-only preprocessing and model parameters. It reads only the test split, predicts on Spark partitions, and writes per-head metrics and partitioned predictions.
