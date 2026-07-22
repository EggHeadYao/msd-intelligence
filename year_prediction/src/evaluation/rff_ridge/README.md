# T90 plus RFF SGD Ridge Evaluation

```bash
spark-submit --master spark://spark-master:7077 \
  year_prediction/src/evaluation/rff_ridge/evaluate.py \
  --model-root parquets/year_prediction/models/t90-rff-sgd-ridge \
  --input parquets/year_prediction/training/t90/vectors.parquet \
  --output parquets/year_prediction/results/rff-ridge
```

