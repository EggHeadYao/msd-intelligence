# PCA SGD Ridge Evaluation

```bash
spark-submit --master spark://spark-master:7077 \
  year_prediction/src/evaluation/pca_ridge/evaluate.py \
  --model-root parquets/year_prediction/models/pca-ridge-t90 \
  --input parquets/year_prediction/training/t90/vectors.parquet \
  --output parquets/year_prediction/results/pca-ridge
```

