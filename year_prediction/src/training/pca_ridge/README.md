# PCA SGD Ridge

This trainer selects the smallest PCA projection reaching 95 percent cumulative
training variance, then optimizes squared loss plus L2 regularization with
distributed gradient descent.

```bash
spark-submit --master spark://spark-master:7077 \
  year_prediction/src/training/pca_ridge/train.py \
  --config year_prediction/config/experiment_a/pca_ridge_t90.json
```

