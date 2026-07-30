# RFF Huber SGD Evaluation

```bash
spark-submit --master spark://spark-master:7077 \
  src/year_prediction/src/evaluation/rff_huber/evaluate.py \
  --model-root parquets/year_prediction/models/rff-huber-sgd \
  --input parquets/year_prediction/training/t90/vectors.parquet \
  --output parquets/year_prediction/results/rff-huber
```

