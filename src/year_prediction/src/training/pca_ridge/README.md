# PCA SGD Ridge

This trainer selects the smallest PCA projection reaching 95 percent cumulative training variance, then optimizes squared loss plus L2 regularization with distributed gradient descent.

```bash
spark-submit --master spark://spark-master:7077 \
  src/year_prediction/src/training/pca_ridge/train.py \
  --config src/year_prediction/config/pca_ridge_t90.json
```
