# Spark LightGBM evaluation

Use the `SYNAPSEML`, `PYSPARK_PYTHON`, and `LD_LIBRARY_PATH` exports from the training guide.

Run artifact validation before opening the fixed test split:

```bash
p1team02/year_prediction/.synapseml-venv/bin/spark-submit \
  --master 'local[2]' \
  --driver-memory 4g \
  --packages "$SYNAPSEML" \
  --conf spark.hadoop.fs.defaultFS=file:/// \
  p1team02/year_prediction/src/evaluation/lightgbm/validate.py \
  --model-root parquets/year_prediction/models/lightgbm-l2-v2
```

Evaluate the frozen model once on test artists:

```bash
p1team02/year_prediction/.synapseml-venv/bin/spark-submit \
  --master 'local[2]' \
  --driver-memory 4g \
  --packages "$SYNAPSEML" \
  --conf spark.hadoop.fs.defaultFS=file:/// \
  p1team02/year_prediction/src/evaluation/lightgbm/evaluate.py \
  --model-root parquets/year_prediction/models/lightgbm-l2-v2 \
  --input parquets/year_prediction/features/full_tabular.parquet \
  --manifest parquets/year_prediction/features/manifest.json \
  --output parquets/year_prediction/results/experiment_a/lightgbm/lightgbm-l2-v2/test \
  --partitions 4
```

The output contains test predictions, MAE/RMSE and decade metrics, plus Spark run metadata. The evaluator rejects a model whose saved 594-feature order hash does not match the input manifest.
