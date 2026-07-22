# T90 plus RFF SGD Ridge

This trainer concatenates the 90 standardized T90 values with 512 deterministic
RBF random Fourier features. It optimizes squared loss plus L2 regularization.
The checked-in configuration records the validation-selected gamma and L2 value.

```bash
spark-submit --master spark://spark-master:7077 \
  year_prediction/src/training/rff_ridge/train.py \
  --config year_prediction/config/experiment_a/rff_ridge_t90.json
```

